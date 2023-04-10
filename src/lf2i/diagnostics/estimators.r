library("mgcv")


fit_gam <- function(indicators, parameters, d) {
  data <- data.frame(indicators, parameters)
  colnames(data) <- c("indicator", paste0("theta", 0:(d - 1)))

  if (d <= 5) {
    # unique tensor
    string_formula <- paste0(
      "indicator ~ s(",
      paste0("theta", 0:(d - 1), collapse = ","),
      ", k = -1, fx = FALSE, bs = \"ts\")"
    )
  } else {
    # one tensor for each variable and use gam additivity; avoid memoryerror
    string_formula <- paste0(
      "indicator ~ ",
      paste0("s(theta", 0:(d - 1),
      ", k = -1, fx = FALSE, bs = \"ts\")", collapse = "+")
    )
  }

  gam_model <- gam(
    formula = as.formula(string_formula),
    method = "REML",
    family = binomial(link = "logit"),
    data = data
  )

  return(list("gam" = gam_model))
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
