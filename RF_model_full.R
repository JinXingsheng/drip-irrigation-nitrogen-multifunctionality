# =========================================================
# Final stable version: Random forest imputation using ranger
# =========================================================

# =========================
# 1. Load packages
# =========================

library(readr)
library(dplyr)
library(ggplot2)
library(tibble)
library(purrr)
library(stringr)
library(patchwork)
library(ranger)
library(scales)

# =========================
# 2. Paths
# =========================

data_dir <- "D:/"
out_dir  <- file.path(data_dir, "RF_Model_Result_Final")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_dir, "01_Imputed_Data"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_dir, "02_Variable_Importance"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_dir, "03_Model_Fit"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_dir, "04_Model_Metrics"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_dir, "05_Variable_Importance_Table"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_dir, "06_CV_Results"), showWarnings = FALSE, recursive = TRUE)

input_file <- "D:/Predicition-full.csv"

message("Current input file: ", input_file)

# =========================
# 3. Colors
# =========================

color_mapping <- c(
  N2O   = "#F5AE6B",
  CH4   = "#A9D179",
  CO2   = "#84CAC0",
  Yield = "#0072B2",
  SOC   = "#F39B7FB2"
)

# =========================
# 4. Read data
# =========================

raw_dat <- read_csv(input_file, show_col_types = FALSE)

# =========================
# 5. Column definitions
# =========================

predictor_cols <- c(
  "Climate.zone", "precipitation.mm", "Temperature",
  "Soil.type", "pH", "SOC", "Clay", "Sandy",
  "BD", "TP", "TN", "Crop.type", "N.addition.rate",
  "Fertilizer.type"
)

factor_cols <- c("Climate.zone", "Soil.type", "Crop.type", "Fertilizer.type")
numeric_cols <- setdiff(predictor_cols, factor_cols)

outcome_map <- list(
  N2O   = list(rr = "N2ORR"),
  CH4   = list(rr = "CH4RR"),
  CO2   = list(rr = "CO2RR"),
  Yield = list(rr = "YieldRR"),
  SOC   = list(rr = "SOCRR")
)

# =========================
# 6. Utility functions
# =========================

get_mode <- function(x) {
  x <- x[!is.na(x) & x != ""]
  if (length(x) == 0) return(NA_character_)
  
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

collapse_rare_levels <- function(x, min_count = 5, other_label = "Other") {
  x <- as.character(x)
  x[is.na(x) | x == ""] <- other_label
  
  tb <- table(x)
  rare_levels <- names(tb)[tb < min_count]
  
  x[x %in% rare_levels] <- other_label
  
  factor(x)
}

align_factor_levels <- function(train_df, valid_df, factor_cols) {
  for (fc in factor_cols) {
    tr <- as.character(train_df[[fc]])
    va <- as.character(valid_df[[fc]])
    
    tr_mode <- get_mode(tr)
    tr_levels <- unique(tr)
    
    tr[is.na(tr) | tr == ""] <- tr_mode
    va[is.na(va) | va == ""] <- tr_mode
    va[!(va %in% tr_levels)] <- tr_mode
    
    train_df[[fc]] <- factor(tr, levels = tr_levels)
    valid_df[[fc]] <- factor(va, levels = tr_levels)
  }
  
  list(train_df = train_df, valid_df = valid_df)
}

align_newdata_to_train <- function(train_df, new_df, factor_cols) {
  for (fc in factor_cols) {
    tr <- as.character(train_df[[fc]])
    nd <- as.character(new_df[[fc]])
    
    tr_mode <- get_mode(tr)
    tr_levels <- unique(tr)
    
    nd[is.na(nd) | nd == ""] <- tr_mode
    nd[!(nd %in% tr_levels)] <- tr_mode
    
    train_df[[fc]] <- factor(tr, levels = tr_levels)
    new_df[[fc]]   <- factor(nd, levels = tr_levels)
  }
  
  list(train_df = train_df, new_df = new_df)
}

drop_zero_var_cols <- function(df, cols) {
  keep <- cols
  
  for (cc in cols) {
    x <- df[[cc]]
    
    if (is.factor(x)) {
      if (length(unique(as.character(x))) <= 1) {
        keep <- setdiff(keep, cc)
      }
    } else {
      x <- as.numeric(x)
      
      if (all(is.na(x)) || sd(x, na.rm = TRUE) == 0) {
        keep <- setdiff(keep, cc)
      }
    }
  }
  
  keep
}

rmse_fun <- function(obs, pred) {
  sqrt(mean((obs - pred)^2, na.rm = TRUE))
}

mae_fun <- function(obs, pred) {
  mean(abs(obs - pred), na.rm = TRUE)
}

r2_fun <- function(obs, pred) {
  ok <- is.finite(obs) & is.finite(pred)
  
  obs <- obs[ok]
  pred <- pred[ok]
  
  if (length(obs) < 3) return(NA_real_)
  if (sd(obs) == 0 || sd(pred) == 0) return(NA_real_)
  
  cor(obs, pred)^2
}

format_p <- function(p) {
  if (is.na(p)) return("p = NA")
  if (p < 0.001) return("p < 0.001")
  
  paste0("p = ", formatC(p, format = "f", digits = 3))
}

format_eq <- function(intercept, slope) {
  sign_txt <- ifelse(intercept >= 0, "+", "-")
  
  paste0(
    "y = ",
    formatC(slope, format = "f", digits = 2),
    " x ",
    sign_txt,
    " ",
    formatC(abs(intercept), format = "f", digits = 2)
  )
}

make_group_folds <- function(study_vec, k = 5, seed = 123) {
  set.seed(seed)
  
  study_vec <- as.character(study_vec)
  u <- unique(study_vec)
  
  n_group <- length(u)
  k_use <- min(k, n_group)
  
  u <- sample(u, length(u))
  fold_id <- rep(1:k_use, length.out = length(u))
  
  folds <- vector("list", k_use)
  
  for (i in seq_len(k_use)) {
    valid_study <- u[fold_id == i]
    folds[[i]] <- which(study_vec %in% valid_study)
  }
  
  folds
}

make_train_valid_split <- function(df, p = 0.75, seed = 1234) {
  set.seed(seed)
  
  u <- unique(as.character(df$study))
  
  if (length(u) >= 4) {
    n_train_group <- max(1, floor(length(u) * p))
    train_group <- sample(u, n_train_group)
    
    train_idx <- which(as.character(df$study) %in% train_group)
    valid_idx <- setdiff(seq_len(nrow(df)), train_idx)
  } else {
    train_idx <- sample(seq_len(nrow(df)), size = floor(nrow(df) * p))
    valid_idx <- setdiff(seq_len(nrow(df)), train_idx)
  }
  
  list(
    train_idx = sort(train_idx),
    valid_idx = sort(valid_idx)
  )
}

safe_fit_ranger <- function(df, vars, mtry, min_node_size, num_trees = 1000, seed = 1) {
  set.seed(seed)
  
  fm <- as.formula(
    paste("yi ~", paste(vars, collapse = " + "))
  )
  
  fit <- tryCatch(
    ranger(
      formula = fm,
      data = df[, c("yi", vars), drop = FALSE],
      num.trees = num_trees,
      mtry = mtry,
      min.node.size = min_node_size,
      importance = "permutation",
      respect.unordered.factors = "order",
      seed = seed
    ),
    error = function(e) NULL
  )
  
  fit
}

safe_predict_ranger <- function(model, newdata) {
  if (is.null(model)) {
    return(rep(NA_real_, nrow(newdata)))
  }
  
  pr <- tryCatch(
    predict(model, data = newdata)$predictions,
    error = function(e) NULL
  )
  
  if (is.null(pr)) {
    return(rep(NA_real_, nrow(newdata)))
  }
  
  as.numeric(pr)
}

manual_cv_ranger <- function(train_df, vars, factor_cols, outcome_name, seed = 123) {
  folds <- make_group_folds(train_df$study, k = 5, seed = seed)
  
  p <- length(vars)
  
  mtry_vals <- unique(
    pmax(1, pmin(c(2, 3, 4, floor(sqrt(p))), p))
  )
  
  node_vals <- c(2, 3, 5, 8)
  
  grid <- expand.grid(
    mtry = mtry_vals,
    min.node.size = node_vals,
    stringsAsFactors = FALSE
  )
  
  out <- vector("list", nrow(grid))
  
  for (g in seq_len(nrow(grid))) {
    gm <- grid$mtry[g]
    gn <- grid$min.node.size[g]
    
    pred_all <- rep(NA_real_, nrow(train_df))
    
    for (i in seq_along(folds)) {
      valid_idx <- folds[[i]]
      train_idx <- setdiff(seq_len(nrow(train_df)), valid_idx)
      
      tr <- train_df[train_idx, , drop = FALSE]
      va <- train_df[valid_idx, , drop = FALSE]
      
      aligned <- align_factor_levels(
        tr,
        va,
        intersect(factor_cols, vars)
      )
      
      tr <- aligned$train_df
      va <- aligned$valid_df
      
      fit_i <- safe_fit_ranger(
        df = tr,
        vars = vars,
        mtry = gm,
        min_node_size = gn,
        num_trees = 1000,
        seed = seed + g * 100 + i
      )
      
      if (is.null(fit_i)) next
      
      pred_i <- safe_predict_ranger(
        fit_i,
        va[, vars, drop = FALSE]
      )
      
      pred_all[valid_idx] <- pred_i
    }
    
    ok <- is.finite(train_df$yi) & is.finite(pred_all)
    
    out[[g]] <- tibble(
      outcome = outcome_name,
      mtry = gm,
      min.node.size = gn,
      n_pred = sum(ok),
      RMSE = ifelse(
        sum(ok) >= 10,
        rmse_fun(train_df$yi[ok], pred_all[ok]),
        NA_real_
      ),
      MAE = ifelse(
        sum(ok) >= 10,
        mae_fun(train_df$yi[ok], pred_all[ok]),
        NA_real_
      ),
      R2 = ifelse(
        sum(ok) >= 10,
        r2_fun(train_df$yi[ok], pred_all[ok]),
        NA_real_
      )
    )
  }
  
  bind_rows(out)
}

# =========================
# 7. Plotting functions
# =========================

plot_varimp <- function(vip_df, outcome_name, n_obs, r2_val, fill_col) {
  vip_df <- vip_df %>%
    arrange(importance_pct) %>%
    mutate(variable = factor(variable, levels = variable))
  
  x_max <- max(vip_df$importance_pct, na.rm = TRUE)
  
  if (!is.finite(x_max) || x_max <= 0) {
    x_max <- 1
  }
  
  ggplot(vip_df, aes(x = importance_pct, y = variable)) +
    geom_col(fill = fill_col, width = 0.72) +
    labs(
      title = outcome_name,
      x = "Variable importance (%)",
      y = NULL
    ) +
    annotate(
      "text",
      x = x_max * 0.95,
      y = 1.05,
      label = paste0(
        "n = ",
        n_obs,
        "\nR² = ",
        formatC(r2_val, format = "f", digits = 2)
      ),
      hjust = 1,
      vjust = 0,
      size = 6
    ) +
    scale_x_continuous(
      expand = expansion(mult = c(0, 0.05))
    ) +
    theme_classic(base_size = 18) +
    theme(
      plot.title = element_text(size = 24, face = "bold", hjust = 0.5),
      axis.title.x = element_text(size = 18, face = "bold"),
      axis.text.x = element_text(size = 16),
      axis.text.y = element_text(size = 16),
      axis.line = element_line(linewidth = 1),
      axis.ticks = element_line(linewidth = 1)
    )
}

plot_fit <- function(fit_df, outcome_name, point_col) {
  lm_fit <- lm(pred ~ obs, data = fit_df)
  
  cf <- coef(lm_fit)
  
  intercept <- unname(cf[1])
  slope <- unname(cf[2])
  
  r2 <- summary(lm_fit)$r.squared
  p_val <- coef(summary(lm_fit))[2, 4]
  
  x_rng <- range(fit_df$obs, na.rm = TRUE)
  y_rng <- range(fit_df$pred, na.rm = TRUE)
  
  x_span <- diff(x_rng)
  y_span <- diff(y_rng)
  
  if (x_span == 0) x_span <- 1
  if (y_span == 0) y_span <- 1
  
  ggplot(fit_df, aes(x = obs, y = pred)) +
    geom_point(
      color = point_col,
      alpha = 0.60,
      size = 3.2
    ) +
    geom_abline(
      slope = 1,
      intercept = 0,
      linetype = "dashed",
      color = "grey50",
      linewidth = 0.8
    ) +
    geom_smooth(
      method = "lm",
      se = FALSE,
      color = "black",
      linewidth = 1.1
    ) +
    labs(
      title = outcome_name,
      x = "Observed",
      y = "Predicted"
    ) +
    annotate(
      "text",
      x = x_rng[1] + 0.07 * x_span,
      y = y_rng[2] - 0.08 * y_span,
      label = format_eq(intercept, slope),
      hjust = 0,
      vjust = 1,
      size = 6.5
    ) +
    annotate(
      "text",
      x = x_rng[2] - 0.05 * x_span,
      y = y_rng[1] + 0.08 * y_span,
      label = paste0(
        "R² = ",
        formatC(r2, format = "f", digits = 2),
        "\n",
        format_p(p_val)
      ),
      hjust = 1,
      vjust = 0,
      size = 6.5
    ) +
    theme_classic(base_size = 18) +
    theme(
      plot.title = element_text(size = 24, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 20, face = "bold"),
      axis.text = element_text(size = 16),
      axis.line = element_line(linewidth = 1),
      axis.ticks = element_line(linewidth = 1)
    )
}

# =========================
# 8. Global data preprocessing
# =========================

dat <- raw_dat

if (!"study" %in% names(dat)) {
  stop("The dataset does not contain the 'study' column.")
}

dat$study <- as.factor(dat$study)

for (cc in factor_cols) {
  dat[[cc]] <- collapse_rare_levels(
    dat[[cc]],
    min_count = 5,
    other_label = "Other"
  )
}

for (cc in numeric_cols) {
  tmp <- as.numeric(dat[[cc]])
  med <- median(tmp, na.rm = TRUE)
  
  tmp[is.na(tmp)] <- med
  
  dat[[cc]] <- tmp
}

# =========================
# 9. Single-outcome modeling
# =========================

run_one_outcome <- function(dat, outcome_name, rr_col, predictor_cols, factor_cols,
                            color_mapping, out_dir, seed_base = 1000) {
  message("\n============================")
  message("Processing outcome: ", outcome_name)
  message("============================")
  
  d0 <- dat
  d0$yi <- as.numeric(d0[[rr_col]])
  
  obs_df <- d0 %>%
    filter(!is.na(yi), is.finite(yi))
  
  n_obs <- nrow(obs_df)
  
  if (n_obs < 20) {
    stop(paste0(outcome_name, " has insufficient valid sample size."))
  }
  
  predictor_use <- drop_zero_var_cols(obs_df, predictor_cols)
  factor_use <- intersect(factor_cols, predictor_use)
  
  split_obj <- make_train_valid_split(
    obs_df,
    p = 0.75,
    seed = seed_base
  )
  
  train_df <- obs_df[split_obj$train_idx, , drop = FALSE]
  valid_df <- obs_df[split_obj$valid_idx, , drop = FALSE]
  
  if (nrow(valid_df) < 5) {
    train_df <- obs_df
    valid_df <- obs_df
  }
  
  predictor_use <- drop_zero_var_cols(train_df, predictor_use)
  factor_use <- intersect(factor_use, predictor_use)
  
  cv_res <- manual_cv_ranger(
    train_df = train_df,
    vars = predictor_use,
    factor_cols = factor_use,
    outcome_name = outcome_name,
    seed = seed_base + 10
  )
  
  write_csv(
    cv_res,
    file.path(
      out_dir,
      "06_CV_Results",
      paste0(outcome_name, "_CV_results.csv")
    )
  )
  
  cv_ok <- cv_res %>%
    filter(is.finite(RMSE), !is.na(RMSE)) %>%
    arrange(RMSE)
  
  if (nrow(cv_ok) == 0) {
    best <- tibble(
      mtry = max(1, min(3, length(predictor_use))),
      min.node.size = 5
    )
  } else {
    best <- cv_ok[1, ]
  }
  
  aligned_tv <- align_factor_levels(
    train_df,
    valid_df,
    factor_use
  )
  
  train_df2 <- aligned_tv$train_df
  valid_df2 <- aligned_tv$valid_df
  
  rf_train <- safe_fit_ranger(
    df = train_df2,
    vars = predictor_use,
    mtry = best$mtry,
    min_node_size = best$min.node.size,
    num_trees = 1500,
    seed = seed_base + 20
  )
  
  if (is.null(rf_train)) {
    stop(paste0(outcome_name, " training model failed."))
  }
  
  valid_pred <- safe_predict_ranger(
    rf_train,
    valid_df2[, predictor_use, drop = FALSE]
  )
  
  valid_eval <- tibble(
    obs = valid_df2$yi,
    pred = valid_pred
  ) %>%
    filter(is.finite(obs), is.finite(pred))
  
  rmse_val <- ifelse(
    nrow(valid_eval) >= 3,
    rmse_fun(valid_eval$obs, valid_eval$pred),
    NA_real_
  )
  
  mae_val <- ifelse(
    nrow(valid_eval) >= 3,
    mae_fun(valid_eval$obs, valid_eval$pred),
    NA_real_
  )
  
  r2_val <- ifelse(
    nrow(valid_eval) >= 3,
    r2_fun(valid_eval$obs, valid_eval$pred),
    NA_real_
  )
  
  aligned_obs <- align_newdata_to_train(
    train_df2,
    obs_df,
    factor_use
  )
  
  obs_df2 <- aligned_obs$new_df
  
  rf_final <- safe_fit_ranger(
    df = obs_df2,
    vars = predictor_use,
    mtry = best$mtry,
    min_node_size = best$min.node.size,
    num_trees = 2000,
    seed = seed_base + 30
  )
  
  if (is.null(rf_final)) {
    stop(paste0(outcome_name, " final model failed."))
  }
  
  obs_pred <- safe_predict_ranger(
    rf_final,
    obs_df2[, predictor_use, drop = FALSE]
  )
  
  fit_df <- tibble(
    obs = obs_df2$yi,
    pred = obs_pred
  ) %>%
    filter(is.finite(obs), is.finite(pred))
  
  full_r2 <- summary(lm(pred ~ obs, data = fit_df))$r.squared
  full_p <- coef(summary(lm(pred ~ obs, data = fit_df)))[2, 4]
  
  aligned_all <- align_newdata_to_train(
    obs_df2,
    d0,
    factor_use
  )
  
  all_df2 <- aligned_all$new_df
  
  pred_all <- safe_predict_ranger(
    rf_final,
    all_df2[, predictor_use, drop = FALSE]
  )
  
  pred_col <- paste0(outcome_name, "Pred")
  full_col <- paste0(outcome_name, "Full")
  
  fill_cols <- tibble(
    !!pred_col := pred_all,
    !!full_col := ifelse(is.na(d0$yi), pred_all, d0$yi)
  )
  
  vip <- sort(rf_final$variable.importance, decreasing = TRUE)
  
  vip_df <- tibble(
    variable = names(vip),
    Overall = as.numeric(vip)
  ) %>%
    mutate(
      Overall = pmax(Overall, 0)
    )
  
  if (sum(vip_df$Overall, na.rm = TRUE) > 0) {
    vip_df <- vip_df %>%
      mutate(
        importance_pct = Overall / sum(Overall, na.rm = TRUE) * 100
      )
  } else {
    vip_df <- vip_df %>%
      mutate(
        importance_pct = 100 / n()
      )
  }
  
  vip_plot <- plot_varimp(
    vip_df = vip_df,
    outcome_name = outcome_name,
    n_obs = n_obs,
    r2_val = full_r2,
    fill_col = color_mapping[outcome_name]
  )
  
  fit_plot <- plot_fit(
    fit_df = fit_df,
    outcome_name = outcome_name,
    point_col = color_mapping[outcome_name]
  )
  
  ggsave(
    file.path(
      out_dir,
      "02_Variable_Importance",
      paste0(outcome_name, "_variable_importance.png")
    ),
    vip_plot,
    width = 7,
    height = 6,
    dpi = 300
  )
  
  ggsave(
    file.path(
      out_dir,
      "03_Model_Fit",
      paste0(outcome_name, "_model_fit.png")
    ),
    fit_plot,
    width = 6.5,
    height = 6,
    dpi = 300
  )
  
  write_csv(
    vip_df,
    file.path(
      out_dir,
      "05_Variable_Importance_Table",
      paste0(outcome_name, "_variable_importance.csv")
    )
  )
  
  metric_df <- tibble(
    Outcome = outcome_name,
    n_obs = n_obs,
    RMSE_valid = rmse_val,
    MAE_valid = mae_val,
    R2_valid = r2_val,
    R2_full = full_r2,
    p_full = full_p,
    mtry = best$mtry,
    min.node.size = best$min.node.size,
    retained_predictors = paste(predictor_use, collapse = "; ")
  )
  
  list(
    fill_cols = fill_cols,
    metric_df = metric_df,
    vip_plot = vip_plot,
    fit_plot = fit_plot
  )
}

# =========================
# 10. Batch modeling
# =========================

results <- list()
all_metrics <- list()
all_vip_plots <- list()
all_fit_plots <- list()

dat_out <- dat
outcome_order <- c("N2O", "CH4", "CO2", "Yield", "SOC")

for (i in seq_along(outcome_order)) {
  out <- outcome_order[i]
  rr_col <- outcome_map[[out]]$rr
  
  res <- run_one_outcome(
    dat = dat_out,
    outcome_name = out,
    rr_col = rr_col,
    predictor_cols = predictor_cols,
    factor_cols = factor_cols,
    color_mapping = color_mapping,
    out_dir = out_dir,
    seed_base = 3000 + i * 100
  )
  
  dat_out <- bind_cols(dat_out, res$fill_cols)
  
  results[[out]] <- res
  all_metrics[[out]] <- res$metric_df
  all_vip_plots[[out]] <- res$vip_plot
  all_fit_plots[[out]] <- res$fit_plot
}

# =========================
# 11. Output results
# =========================

metric_table <- bind_rows(all_metrics)

write_csv(
  metric_table,
  file.path(
    out_dir,
    "04_Model_Metrics",
    "RF_model_metrics_summary.csv"
  )
)

write_csv(
  dat_out,
  file.path(
    out_dir,
    "01_Imputed_Data",
    "Prediction_full_with_RF.csv"
  )
)

pred_full_cols <- c(
  "N2OPred", "N2OFull",
  "CH4Pred", "CH4Full",
  "CO2Pred", "CO2Full",
  "YieldPred", "YieldFull",
  "SOCPred", "SOCFull"
)

pred_full_cols <- pred_full_cols[pred_full_cols %in% names(dat_out)]

write_csv(
  dat_out[, pred_full_cols, drop = FALSE],
  file.path(
    out_dir,
    "01_Imputed_Data",
    "Only_Pred_and_Full.csv"
  )
)

vip_combined <- (
  all_vip_plots[["N2O"]] | all_vip_plots[["CH4"]] | all_vip_plots[["CO2"]]
) / (
  all_vip_plots[["Yield"]] | all_vip_plots[["SOC"]] | patchwork::plot_spacer()
)

fit_combined <- (
  all_fit_plots[["N2O"]] | all_fit_plots[["CH4"]] | all_fit_plots[["CO2"]]
) / (
  all_fit_plots[["Yield"]] | all_fit_plots[["SOC"]] | patchwork::plot_spacer()
)

ggsave(
  file.path(
    out_dir,
    "02_Variable_Importance",
    "All_variable_importance_combined.png"
  ),
  vip_combined,
  width = 14,
  height = 10,
  dpi = 300
)

ggsave(
  file.path(
    out_dir,
    "03_Model_Fit",
    "All_model_fit_combined.png"
  ),
  fit_combined,
  width = 14,
  height = 10,
  dpi = 300
)
