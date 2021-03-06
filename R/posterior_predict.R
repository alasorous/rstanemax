#' Outcome prediction from posterior distribution of parameters
#'
#' Compute outcome predictions using posterior samples.
#' Exposure data for prediction can be either original data used for model fit or new data.
#'
#' Run \code{vignette("emaxmodel", package = "rstanemax")} to see
#' how you can use the posterior prediction for plotting estimated Emax curve.
#'
#' @export
#' @export posterior_predict
#' @name posterior_predict
#' @param object A `stanemax` class object
#' @param newdata An optional data frame that contains columns needed for model to run (exposure and covariates).
#' If the model does not have any covariate, this can be a numeric vector corresponding to the exposure metric.
#' @param returnType An optional string specifying the type of return object.
#' @param newDataType An optional string specifying the type of newdata input,
#' whether in the format of an original data frame or a processed model frame.
#' Mostly used for internal purposes and users can usually leave at default.
#' @param ... Additional arguments passed to methods.
#' @return An object that contain predicted response with posterior distribution of parameters.
#' The default is a matrix containing predicted response.
#' Each row of the matrix is a vector of predictions generated using a single draw of the model parameters from the posterior distribution.
#'
#' If either _dataframe_ or _tibble_ is specified, the function returns a data frame or tibble object in a long format -
#' each row is a prediction generated using a single draw of the model parameters and a corresponding exposure.
#'
#' Two types of predictions are generated with this function.
#' _respHat_ corresponds to the prediction without considering residual variability and is intended to provide credible interval of "mean" response.
#' _response_ include residual variability in its calculation, therefore the range represents prediction interval of observed response.
#'
#' The return object also contains exposure and parameter values used for calculation.
#'
posterior_predict.stanemax <- function(object, newdata = NULL,
                                       returnType = c("matrix", "dataframe", "tibble"),
                                       newDataType = c("raw", "modelframe"),
                                       ...){

  returnType <- match.arg(returnType)
  newDataType <- match.arg(newDataType)


  if(is.null(newdata)) {
    df.model <- object$prminput$df.model
  } else {
    if(newDataType == "modelframe"){
      df.model <- newdata
    } else {
      if(is.vector(newdata)) {
        newdata <- data.frame(newdata)
        names(newdata) <- as.character(object$prminput$formula[[3]])
      }

      df.model <- create_model_frame(formula = object$prminput$formula,
                                     data = newdata,
                                     param.cov = object$prminput$param.cov,
                                     cov.levels = object$prminput$cov.levels,
                                     is.model.fit = FALSE)
    }
  }




  pred.response.raw <- pp_calc(object$stanfit, df.model)

  cov.fct.numeric <-
    df.model %>%
    dplyr::select(covemaxfct = covemax,
                  covec50fct = covec50,
                  cove0fct   = cove0) %>%
    dplyr::distinct() %>%
    dplyr::mutate(covemax = as.numeric(covemaxfct),
                  covec50 = as.numeric(covec50fct),
                  cove0   = as.numeric(cove0fct))

  pred.response <-
    dplyr::left_join(pred.response.raw, cov.fct.numeric, by = c("covemax", "covec50", "cove0")) %>%
    dplyr::select(-(covemax:cove0)) %>%
    dplyr::select(mcmcid, exposure,
                  covemax = covemaxfct,
                  covec50 = covec50fct,
                  cove0   = cove0fct,
                  dplyr::everything())

  if(returnType == "matrix") {
    return(matrix(pred.response$response, ncol = nrow(df.model), byrow = TRUE))
  } else if(returnType == "dataframe") {
    return(as.data.frame(pred.response))
  } else if(returnType == "tibble") {
    return(dplyr::as_tibble(pred.response))
  }

}


# Calculate posterior prediction from stanfit object and exposure data
## data.pp is a data frame with column named `exposure`
pp_calc <- function(stanfit, df.model){

  param.fit <- extract_param_fit(stanfit)

  df <-
    df.model %>%
    dplyr::mutate(covemax = as.numeric(covemax),
                  covec50 = as.numeric(covec50),
                  cove0   = as.numeric(cove0)) %>%
    tidyr::expand_grid(mcmcid = 1:max(param.fit$mcmcid), .) %>%
    dplyr::left_join(param.fit, by = c("mcmcid", "covemax", "covec50", "cove0"))

  out <-
    df %>%
    dplyr::mutate(respHat = e0 + emax * exposure^gamma / (ec50^gamma + exposure^gamma),
                  response= stats::rnorm(respHat, respHat, sigma)) %>%
    dplyr::select(mcmcid, exposure, dplyr::everything())

}


extract_param_fit <- function(stanfit){

  param.extract <- c("emax", "e0", "ec50", "gamma", "sigma")

  param.extract.1 <- rstan::extract(stanfit, pars = c("emax", "e0", "ec50"))
  param.extract.2 <- rstan::extract(stanfit, pars = c("gamma", "sigma"))

  extract_params_covs <- function(k){
    out <-
      dplyr::as_tibble(param.extract.1[[k]]) %>%
      dplyr::mutate(mcmcid = dplyr::row_number()) %>%
      tidyr::pivot_longer(-mcmcid,
                          names_to = paste0("cov", k),
                          values_to = k,
                          names_prefix = "V")
  }

  param.fit.withcov <-
    dplyr::full_join(extract_params_covs("emax"),
                     extract_params_covs("e0"), by = "mcmcid") %>%
    dplyr::full_join(extract_params_covs("ec50"), by = "mcmcid") %>%
    dplyr::mutate(covemax = as.numeric(covemax),
                  covec50 = as.numeric(covec50),
                  cove0   = as.numeric(cove0))

  param.fit  <-
    param.extract.2 %>%
    dplyr::as_tibble() %>%
    dplyr::mutate(mcmcid = dplyr::row_number()) %>%
    dplyr::full_join(param.fit.withcov, ., by = "mcmcid")

  return(param.fit)
}


#' @rdname posterior_predict
#' @export
#' @param ci Credible interval of the response without residual variability.
#' @param pi Prediction interval of the response with residual variability.
#'
posterior_predict_quantile <- function(object, newdata = NULL, ci = 0.9, pi = 0.9,
                                       newDataType = c("raw", "modelframe")){

  pp.raw <- posterior_predict.stanemax(object, newdata,
                                       returnType = c("tibble"), newDataType = newDataType)

  ndata <-
    dplyr::filter(pp.raw, mcmcid == 1) %>%
    nrow()

  pp.raw.2 <-
    pp.raw %>%
    dplyr::mutate(dataid = rep(1:ndata, length.out = nrow(.)))

  pp.quantile <-
    pp.raw.2 %>%
    dplyr::group_by(exposure, covemax, covec50, cove0, Covariates, dataid) %>%
    dplyr::summarize(ci_low = stats::quantile(respHat, probs = 0.5 - ci/2),
                     ci_med = stats::quantile(respHat, probs = 0.5),
                     ci_high= stats::quantile(respHat, probs = 0.5 + ci/2),
                     pi_low = stats::quantile(response, probs = 0.5 - pi/2),
                     pi_med = stats::quantile(response, probs = 0.5),
                     pi_high= stats::quantile(response, probs = 0.5 + pi/2)) %>%
    dplyr::arrange(dataid) %>%
    dplyr::ungroup() %>%
    dplyr::select(-dataid)
}




