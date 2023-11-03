library("mgcv")


fit_joint_splines <- function(indicators, parameters, d) {
  data <- data.frame(indicators, parameters)
  colnames(data) <- c("indicator", paste0("theta", 0:(d - 1)))

  # joint tensor product basis
  string_formula <- paste0(
    "indicator ~ s(",
    paste0("theta", 0:(d - 1), collapse = ","),
    ", k = -1, fx = FALSE, bs = \"ts\")"
  )

  gam_model <- gam(
    formula = as.formula(string_formula),
    method = "REML",
    family = binomial(link = "logit"),
    data = data
  )

  return(list("splines" = gam_model))
}


fit_additive_splines <- function(indicators, parameters, d) {
  data <- data.frame(indicators, parameters)
  colnames(data) <- c("indicator", paste0("theta", 0:(d - 1)))

  # one tensor for each variable and use gam additivity
  # avoid memoryerror if joint tensor is too big
  string_formula <- paste0(
    "indicator ~ ",
    paste0("s(theta", 0:(d - 1),
    ", k = -1, fx = FALSE, bs = \"ts\")", collapse = "+")
  )

  gam_model <- gam(
    formula = as.formula(string_formula),
    method = "REML",
    family = binomial(link = "logit"),
    data = data
  )

  return(list("gam_splines" = gam_model))
}


predict_gam <- function(gam_model, parameters, d) {
  newdata <- data.frame(parameters)
  colnames(newdata) <- paste0("theta", 0:(d - 1))
  predictions <- predict(
    gam_model,
    newdata = newdata,
    se.fit = TRUE,
    type = "response"
  )

  return(list(
    "predictions" = predictions$fit,
    "se" = predictions$se.fit
  ))
}
