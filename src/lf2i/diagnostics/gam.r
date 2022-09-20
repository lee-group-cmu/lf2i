library("mgcv")

# TODO: modify this function (and the parent) so that it does not need to load csv files

helper_function <- function(is_azure, d){
  if (is_azure == 'yes') {
    data <- read.csv("/home/azureuser/lmassera/waldo/muons/gam_diagnostics.csv")  
  } else {
    data <- read.csv("/Users/luca/Desktop/uni/cmu/research/waldo/muons/gam_diagnostics.csv")
  }
  
  if (d == 1) {
    gam_diagnostics <- gam(formula = w ~ s(theta, k = -1, fx = FALSE, bs = "ts"), 
                         method="REML",
                         family = binomial(link="logit"), 
                         data = data)
    predictions <- predict(gam_diagnostics, newdata = data.frame("theta" = data$theta), se.fit = TRUE, 
                           type = "response")
  } else {  # main effects plus interaction term; s(theta0, k = -1, fx = FALSE, bs = "cr") + s(theta1, k = -1, fx = FALSE, bs = "cr") +
    gam_diagnostics <- gam(formula = w ~ s(theta0, theta1, k = -1, fx = FALSE, bs = "ts"),
                           method="REML",
                           family = binomial(link="logit"), 
                           data = data)
    predictions <- predict(gam_diagnostics, newdata = data.frame("theta0" = data$theta0, "theta1" = data$theta1), se.fit = TRUE, 
                           type = "response")
  }
  
  
  
  return(list("predictions"=predictions$fit, "se"=predictions$se.fit))
}
