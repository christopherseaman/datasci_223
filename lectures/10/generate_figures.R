# Generate static figures for A/B Testing lecture (base R only)
# Run from lectures/10/ directory

media_dir <- "media"
dir.create(media_dir, showWarnings = FALSE)

# 1. CLT Example (from lines 313-344)
png(file.path(media_dir, "clt_example.png"), width = 800, height = 600, res = 100)
par(mfrow = c(2, 2))

n <- c(25, 100, 800)
wts <- c(0.6, 0.35, 0.05)
means <- c(65, 80, 93)
sds <- c(5, 2.5, 1)

# Population density
x <- seq(50, 100, 0.01)
density <- rowSums(mapply(function(w, m, s) w * dnorm(x, m, s), wts, means, sds))
plot(x, density, type = 'l', main = 'Population Density', xlab = 'x', ylab = 'Density')

rtrimodal <- function(n) {
  component <- sample(1:3, n, replace = TRUE, prob = wts)
  rnorm(n, mean = means[component], sd = sds[component])
}

set.seed(42)
for (nn in n) {
  hist(
    replicate(10000, mean(rtrimodal(nn))),
    breaks = 20,
    xlab = expression(bar(x)),
    main = paste0('Sampling Distribution with N = ', nn),
    xlim = c(50, 100)
  )
}
dev.off()
cat("Generated: clt_example.png\n")

# 2. CI Diagram (from lines 365-390)
png(file.path(media_dir, "ci_diagram.png"), width = 600, height = 150, res = 100)

est <- 0.12
lo <- 0.04
hi <- 0.20

par(mar = c(3, 1, 1, 1))

plot(
  x = c(lo, est, hi), y = c(1, 1, 1),
  xlim = c(lo - 0.05, hi + 0.05), ylim = c(0.97, 1.05),
  xlab = "", ylab = "", yaxt = "n", bty = "n",
  yaxs = "i", pch = NA
)

segments(lo, 1, hi, 1, col = "darkgreen", lwd = 4, lend = "butt")
segments(lo, 0.993, lo, 1.007, col = "darkgreen", lwd = 3)
segments(hi, 0.993, hi, 1.007, col = "darkgreen", lwd = 3)
points(est, 1, pch = 21, bg = "darkgreen", col = "darkgreen", cex = 1.8)

text(est, 1.025, "point estimate", adj = 0.5, cex = 1.25)
text(lo, 1.025, "limit", adj = 0.5, cex = 1.25)
text(hi, 1.025, "limit", adj = 0.5, cex = 1.25)

dev.off()
cat("Generated: ci_diagram.png\n")

# 3. Lift CI Plot (base R version)
png(file.path(media_dir, "lift_ci.png"), width = 400, height = 400, res = 100)

# Using values from lecture example
ybar_c <- 0.0527
ybar_t <- 0.0550
se_c <- 0.0012
se_t <- 0.0012

lift <- ybar_t / ybar_c - 1
var_lift <- (ybar_t / ybar_c)^2 * (se_t^2 / ybar_t^2 + se_c^2 / ybar_c^2)
se_lift <- sqrt(var_lift)

z <- qnorm(0.975)
ci_lo <- lift - z * se_lift
ci_hi <- lift + z * se_lift

par(mar = c(4, 5, 3, 2))
plot(1, lift, xlim = c(0.5, 1.5), ylim = c(min(ci_lo, 0) - 0.02, max(ci_hi, 0) + 0.02),
     pch = 19, cex = 1.5, xaxt = "n", xlab = "", ylab = "Lift",
     main = "Estimated lift with 95% CI")
arrows(1, ci_lo, 1, ci_hi, angle = 90, code = 3, length = 0.1, lwd = 2)
abline(h = 0, lty = 2, col = "grey50")
axis(1, at = 1, labels = "Treatment vs.\nControl", tick = FALSE)
axis(2, at = pretty(c(ci_lo, ci_hi)), labels = paste0(round(pretty(c(ci_lo, ci_hi)) * 100, 1), "%"))

dev.off()
cat("Generated: lift_ci.png\n")

# 4. CUPED Illustration (base R version)
png(file.path(media_dir, "cuped_comparison.png"), width = 500, height = 400, res = 100)

set.seed(42)
n_per_group <- 5000
n <- 2 * n_per_group
rho <- 0.6

# Draw correlated standard normals using Cholesky
z1 <- rnorm(n)
z2 <- rho * z1 + sqrt(1 - rho^2) * rnorm(n)

# Transform to revenue scale (mean=50, sd=20)
pre_revenue <- 50 + 20 * z1
revenue <- 50 + 20 * z2

# Add treatment lift
variant <- rep(c("control", "treatment"), each = n_per_group)
revenue <- revenue + ifelse(variant == "treatment", 2, 0)

# Calculate standard SE
se_ctrl <- sd(revenue[1:n_per_group]) / sqrt(n_per_group)
se_trt <- sd(revenue[(n_per_group+1):n]) / sqrt(n_per_group)
naive <- sqrt(se_ctrl^2 + se_trt^2)

# CUPED adjustment
theta <- cov(revenue, pre_revenue) / var(pre_revenue)
y_adj <- revenue - theta * (pre_revenue - mean(pre_revenue))

se_ctrl_adj <- sd(y_adj[1:n_per_group]) / sqrt(n_per_group)
se_trt_adj <- sd(y_adj[(n_per_group+1):n]) / sqrt(n_per_group)
cuped_se <- sqrt(se_ctrl_adj^2 + se_trt_adj^2)

reduction <- round((1 - cuped_se / naive) * 100, 1)

par(mar = c(4, 5, 3, 2))
bp <- barplot(c(naive, cuped_se), names.arg = c("Standard", "CUPED-adjusted"),
        col = c("grey60", "steelblue"), ylab = "SE of treatment effect",
        main = paste0("CUPED reduces SE by ~", reduction, "%"),
        ylim = c(0, max(naive) * 1.1))

dev.off()
cat("Generated: cuped_comparison.png\n")

cat("\nAll figures generated in", media_dir, "\n")
