library(glmnet)
library(mboost)
library(Matrix)
library(parallel)
library(stabs)
library(InvariantCausalPrediction)
library(tidyverse)

#GLOBAL PARAMETERS
#update these values 
MAINDATA_FILEPATH <- "path/to/mainfile.csv"
LONGDATA_FILEPATH <- "path/to/longfile.csv"
METADATA_FILEPATH <- "path/to/metafile.csv"
OUTPREFIX <- "path/to/outdir/prefix"

# if using defined groups
#EXP_INDICES_FILEPATH <- "path/to/expindices.csv"
YEAR_SPLIT <- 0.6

# get training and target tasks from metafile
meta_df <- read.csv(METADATA_FILEPATH)
patient_col <- meta_df %>% filter(column_type == 'patient_id') %>% pull(column_name) %>% .[1]
target_tasks <- meta_df %>% filter(column_type == 'target_task') %>% pull(column_name)
training_tasks <- meta_df %>% filter(column_type == 'task_label') %>% pull(column_name)
all_tasks <- c(target_tasks, training_tasks)

#all_tasks <- sub("^task", "", all_tasks) #use with example data set

#load longitudinal data and get years for exposure and outcome
long_df <- read.csv(LONGDATA_FILEPATH)
all_years <- sort(unique(long_df$EVENT_YEAR))
num_exposure_years <- floor(length(all_years) * YEAR_SPLIT)
exposure_years <- all_years[1:num_exposure_years]
outcome_years <- all_years[(num_exposure_years + 1):length(all_years)]

# get a list of patients in each task cohort (defined as task not observed in exposure years)
main_df <- read.csv(MAINDATA_FILEPATH)
all_patients <- main_df %>% pull(!!sym(patient_col))

cohorts <- list()
for (task in all_tasks) {
  exclude_patients <- long_df %>% 
    filter(ENDPOINT == task & EVENT_YEAR %in% exposure_years) %>% 
    pull(PATIENT_ID)
  cohorts[[task]] <- all_patients[!all_patients %in% exclude_patients]
}

# Filter out tasks with no longitudinal data available
valid_tasks_indices <- sapply(all_tasks, function(task) {
  any(long_df$ENDPOINT == task)
})
valid_tasks <- all_tasks[valid_tasks_indices]
nonvalid_tasks <- setdiff(all_tasks, valid_tasks)

#Valid and nonvalid tasks are handled separately in icp function
cat("List of valid tasks:", valid_tasks, "\n")
cat("List of nonvalid tasks:", nonvalid_tasks, "\n")

#Get experimental indices
#exp_indices_df <- read.csv(EXP_INDICES_FILEPATH)
exp_indices_df <- ""

aggregate_long_data <- function(df, patient_list) {
  df <- df %>%
    group_by(PATIENT_ID, ENDPOINT) %>%
    summarise(count = n(), .groups = 'drop') %>%
    pivot_wider(names_from = ENDPOINT, values_from = count, values_fill = list(count = 0))
  
  patient_df <- data.frame(PATIENT_ID = patient_list)
  df <- right_join(patient_df, df, by = "PATIENT_ID")
  
  df[is.na(df)] <- 0
  rownames(df) <- df$PATIENT_ID
  df$PATIENT_ID <- NULL
  return(df)
}

get_dataset <- function(long_df, cohorts, valid_tasks, exposure_years, outcome_years, aggregate_long_data, outcome_endpoint) {
  
  if (is.list(outcome_endpoint)) {
    # multiple tasks in outcome
    # cohort corresponds to intersection of cohorts for each task
    task_cohorts <- lapply(outcome_endpoint, function(task) cohorts[[task]])
    task_cohort <- Reduce(intersect, task_cohorts)
    outcome_endpoint_list <- outcome_endpoint
  } else {
    # single task in outcome
    task_cohort <- cohorts[[outcome_endpoint]]
    outcome_endpoint_list <- list(outcome_endpoint)
  }
  
  # get exposure data for patients in task cohort
  data <- long_df %>% filter(ENDPOINT %in% valid_tasks & EVENT_YEAR %in% exposure_years)
  if (nrow(data) == 0) {
    warning("Warning: no data found for the specified tasks and exposure years.")
    return(NULL)
  }
  data <- aggregate_long_data(data, task_cohort)
  
  # check all tasks are in exposures (if a column is missing it means there is no longitudinal data available, raise warning and fill with zeros)
  for (task in valid_tasks) {
    if (!(task %in% colnames(data))) {
      warning(paste("Warning: there is no longitudinal data available for exposure", task, ", removing task"))
      data[[task]] <- NULL
    }
  }
  
  # Update all_tasks to reflect the valid tasks after removing those with no data
  updated_valid_tasks <- colnames(data)
  valid_tasks <- valid_tasks[valid_tasks %in% updated_valid_tasks]
  
  data <- data[ , valid_tasks, drop = FALSE] # make sure columns are in correct order
  
  # construct outcome variable (binary indicator)
  outcome_data <- long_df %>% filter(ENDPOINT %in% outcome_endpoint_list & EVENT_YEAR %in% outcome_years)
  
  if (nrow(outcome_data) == 0) {
    warning("Warning: No outcome data found for the specified tasks and outcome years.")
    return(NULL)
  }
  
  outcome_data <- aggregate_long_data(outcome_data, task_cohort)
  
  # if outcome constructed from multiple tasks, combine into one indicator variable
  outcome_data$outcome_variable <- rowSums(outcome_data)
  
  # merge exposures and outcomes into one dataset
  # first N columns are exposures and (N+1)th column is the outcome
  colnames(data) <- paste0("exposure_", colnames(data))
  data$outcome <- 0
  common_patients <- intersect(rownames(data), rownames(outcome_data))
  data[common_patients, "outcome"] <- outcome_data[common_patients, "outcome_variable"]
  data$outcome <- as.integer(data$outcome >0)
  
  if (ncol(data) != length(valid_tasks) + 1) {
    stop("The number of columns in the data should correspond to the number of exposures (tasks) + 1 for the outcome.")
  }
  
  return(data)
}

cosine_distance <- function(x) {
  # Calculate the dot product
  dot_prod <- x %*% t(x)
  
  # Calculate the magnitude of each vector
  magnitude <- sqrt(rowSums(x^2))
  
  # Calculate cosine similarity
  cos_sim <- dot_prod / (magnitude %*% t(magnitude))
  
  # Convert similarity to distance
  cos_dist <- 1 - cos_sim
  
  # Ensure the diagonal is exactly 0 (sometimes there are small floating point errors)
  diag(cos_dist) <- 0
  
  return(cos_dist)
}

calculate_distances <- function(task_vectors, all_tasks) {
  # Combine vectors into a matrix
  task_matrix <- do.call(rbind, task_vectors)
  rownames(task_matrix) <- all_tasks
  
  # Calculate the distance matrix
  dist_matrix <- cosine_distance(task_matrix)
  
  # Convert to a long-format data frame
  dist_data <- as.data.frame(as.table(dist_matrix))
  colnames(dist_data) <- c("task1", "task2", "value")
  
  return(dist_data)
}

                           
get_icp <- function(outprefix, long_df, cohorts, all_tasks, valid_tasks, exposure_years, outcome_years, aggregate_long_data, exp_indices_df, alpha = 0.1) {
  outpath_dist <- paste0(outprefix, "_causal_distance_ICP_random.csv")
  outpath_est <- paste0(outprefix, "_causal_estimates_ICP_random.csv")
  
  results_df <- data.frame()
  
  cat("Running ICP method... \n")
  for (task in valid_tasks) {
    cat("Processing task:", task, "\n")
    task_data <- get_dataset(long_df, cohorts, valid_tasks, exposure_years, outcome_years, aggregate_long_data, task)
    
    if (is.null(task_data)) {
      next
    }
    X <- as.matrix(task_data %>% select(starts_with("exposure_")))
    y <- task_data$outcome
    y <- factor(y, levels = c(0, 1)) #factor for binary data
    n <- nrow(task_data)

    #Experiment indices, using random allocation
    set.seed(123)  
    random_indices <- sample(n)
    ExpInd <- list()
    ExpInd[[1]] <- random_indices[1:floor(n/2)]
    ExpInd[[2]] <- random_indices[(floor(n/2) + 1):n]

    #Predetermined experiment indices, filter and sort the exp_indices to match task_data
    # patient_ids <- rownames(task_data)
    # task_exp_indices_df <- exp_indices_df %>%
    #   filter(PATIENT_ID %in% patient_ids) %>%
    #   arrange(match(PATIENT_ID, patient_ids))
    
    #ExpInd <- list()
    #ExpInd[[1]] <- which(task_exp_indices_df$Group == 1)
    #ExpInd[[2]] <- which(task_exp_indices_df$Group == 2)
    #ExpInd[[3]] <- which(task_exp_indices_df$Group == 3)
    #ExpInd[[4]] <- which(task_exp_indices_df$Group == 4)
    cat("Number of patients", n, "\n")
    cat("Number of patients in ExpInd for task", task, ":\n")
    cat("Group 1:", length(ExpInd[[1]]), "\n")
    cat("Group 2:", length(ExpInd[[2]]), "\n")
    #cat("Group 3:", length(ExpInd[[3]]), "\n")
    #cat("Group 4:", length(ExpInd[[4]]), "\n")
    cat("\n")
    
    n_samples <- nrow(X)
    y_counts <- table(y)
    cat("Number of samples in X:", n_samples, "\n")
    cat("Count of 0s in y:", y_counts[1], "\n")
    cat("Count of 1s in y:", y_counts[2], "\n")

    #ICP, change preselection if needed (all for no preselection)
    icp <- ICP(X, y, ExpInd, alpha, selection= "boosting", maxNoVariables = 10, maxNoVariablesSimult = 5, showAcceptedSets = TRUE, showCompletion = FALSE) 

    print("Used variables")
    print(icp$usedvariables)

    #List of causal point estimates
    coeff_list <- icp$Coeff
    point_estimate <- numeric(length(coeff_list))

    #Using mean of point estimates in the causal distance
    for (i in seq_along(coeff_list)) {
      point_estimate[i] <- mean(coeff_list[[i]], na.rm = TRUE)
    }
    
    task_df <- data.frame(exposure = valid_tasks, outcome = task, value = point_estimate)
    task_df$value[is.na(task_df$value)] <- 0

    print(task_df)
    
    print("P values")
    print(icp$pvalues)

    results_df <- bind_rows(results_df, task_df)
    cat("\n") 
  }
  
  #add non-valid tasks to the dataframe, set to zero
  non_valid_tasks <- setdiff(all_tasks, valid_tasks)
  
  for (non_valid_task in non_valid_tasks) {
    #Non-valid task as outcome
    cat("Adding novalid task:", non_valid_task, "\n")
    zero_outcome_estimates <- data.frame(
      exposure = all_tasks,
      outcome = non_valid_task,
      value = rep(0, length(all_tasks))
    )
    results_df <- bind_rows(results_df, zero_outcome_estimates)
    
    #Non-valid task as exposure, case where task is both outcome and exposure already added in previous step
    zero_exposure_estimates <- data.frame(
      exposure = non_valid_task,
      outcome = setdiff(all_tasks, non_valid_task),
      value = rep(0, length(all_tasks) -1)
    )
    results_df <- bind_rows(results_df, zero_exposure_estimates)
  }

  #Building task vectors for each task 
  task_vectors <- list()
  for (task in all_tasks) {
    task_vector <- sapply(all_tasks, function(exposure) {
      row_index <- which(results_df$outcome == task & results_df$exposure == exposure)
      return(results_df$value[row_index])
    })
    task_vectors[[task]] <- unlist(task_vector)
  }

  #building the causal distance matrix and table
  cat("Calculating causal distances... \n")
  dist_data <- calculate_distances(task_vectors, all_tasks)
  
  print(dist_data)
    
  #Save the causal distance table to the outpath
  write_csv(dist_data, outpath_dist)
  write_csv(results_df, outpath_est)
}

#Run the method
get_icp(OUTPREFIX, long_df, cohorts, all_tasks, valid_tasks, exposure_years, outcome_years, aggregate_long_data, exp_indices_df)