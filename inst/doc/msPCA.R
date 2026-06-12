## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(collapse = TRUE, comment = "#>")

## ----install, eval=FALSE------------------------------------------------------
# install.packages("msPCA")

## ----load---------------------------------------------------------------------
library(msPCA)

## -----------------------------------------------------------------------------
Sigma <- cor(datasets::mtcars)

set.seed(42)
res <- mspca(Sigma, r = 2, ks = c(4, 4), feasibilityConstraintType = 0, verbose = FALSE)
print_mspca(res, Sigma)

## -----------------------------------------------------------------------------
res_corr <- mspca(Sigma, r = 2, ks = c(4, 4), feasibilityConstraintType = 1, verbose = FALSE)
print_mspca(res_corr, Sigma)

## -----------------------------------------------------------------------------
cat("Diagnostics for res (feasibilityConstraintType = 0)\n")
feasibility_violation_off(Sigma, res$x_best, feasibilityConstraintType = 0)
feasibility_violation_off(Sigma, res$x_best, feasibilityConstraintType = 1)
fraction_variance_explained(Sigma, res$x_best)
fraction_variance_explained_perPC(Sigma, res$x_best)

cat("\nDiagnostics for res_corr (feasibilityConstraintType = 1)\n")
feasibility_violation_off(Sigma, res_corr$x_best, feasibilityConstraintType = 0)
feasibility_violation_off(Sigma, res_corr$x_best, feasibilityConstraintType = 1)
fraction_variance_explained(Sigma, res_corr$x_best)
fraction_variance_explained_perPC(Sigma, res_corr$x_best)

## -----------------------------------------------------------------------------
pca_res <- prcomp(datasets::mtcars, scale. = TRUE)
fraction_variance_explained(Sigma, pca_res$rotation[, 1:2])

