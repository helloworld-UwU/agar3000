

# List of required packages
required_packages <- c(
    "ggplot2",   # plotting
    "dplyr",     # data manipulation
    "tidyr",     # data reshaping
    "tibble",    # enhanced data frames
    "stringr",   # string manipulation
    "forcats",   # factor handling
    "mcr",       # Deming regression
    "DescTools", # CCC calculation
    "rmarkdown", # report rendering
    "argparse"   # command line argument parsing (needs Python)
)
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Install missing packages
for (pkg in required_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        install.packages(pkg, dependencies = TRUE)
    }
    library(pkg, character.only = TRUE)
}

### ====================================================
### FUNCTIONS

# --- 1. RELATIVE BLAND-ALTMAN + sMAPE FUNCTION ---

analyze_relative_ba_sMAPE <- function(data, m1_col, m2_col, title) {
    
    df <- data %>%
        mutate(
            M1 = .data[[m1_col]],
            M2 = .data[[m2_col]],
            # Average (X-axis)
            Average = (M1 + M2) / 2  
        ) %>%
        # CRITICAL FIX: Filter out rows where the average is zero to prevent division by zero (NaN)
        filter(Average > 0) %>% 
        mutate(
            # Relative Difference (Y-axis: Percentage difference relative to the average)
            Percent_Difference = ((M1 - M2) / Average) * 100
        )
    
    # Calculate statistics
    bias_percent <- mean(df$Percent_Difference)
    sd_percent <- sd(df$Percent_Difference)
    loa_upper_percent <- bias_percent + 1.96 * sd_percent
    loa_lower_percent <- bias_percent - 1.96 * sd_percent
    
    # Calculate Symmetric Mean Absolute Percentage Error (sMAPE)
    sMAPE <- mean(abs(df$Percent_Difference))
    
    # Plotting (Relative Bland-Altman Plot) 
    p <- ggplot(df, aes(x = Average, y = Percent_Difference)) +
        geom_point(alpha = 0.6, color = "#2c3e50") +
        #geom_text(aes(label = Plate_ID), vjust = -0.7, size = 3, color = "#2c3e50") +
        geom_hline(yintercept = bias_percent, linetype = "solid", color = "#27ae60", linewidth = 1.2) +
        geom_hline(yintercept = loa_upper_percent, linetype = "dashed", color = "#c0392b", linewidth = 1) +
        geom_hline(yintercept = loa_lower_percent, linetype = "dashed", color = "#c0392b", linewidth = 1) +
        labs(
            title = paste("Relative Bland-Altman Plot:", title),
            subtitle = paste0("Proportional Bias: ", round(bias_percent, 2), "% | sMAPE: ", round(sMAPE, 2), "% (N=", nrow(df), ")"),
            x = "Average Count (Colonies)",
            y = "Percentage Difference (%)"
        ) +
        theme_minimal() +
        theme(plot.title = element_text(face = "bold")) +
        coord_cartesian(ylim = c(-200, 200))
    
    print(p)
    
    # Return metrics
    list(
        bias_percent = bias_percent,
        sMAPE = sMAPE,
        loa_upper_percent = loa_upper_percent,
        loa_lower_percent = loa_lower_percent
    )
}

# --- 2. DEMING REGRESSION FUNCTION ---

# Function to perform Deming Regression (Program vs. Human Reference)
analyze_deming_regression <- function(data, program_col, ref_col, title) {
    
    if (!all(c(program_col, ref_col) %in% colnames(data))) {
        stop("One or both specified columns do not exist in the data frame.")
    }
    
    df <- data %>%
        mutate(
            Program   = as.numeric(.data[[program_col]]),
            Reference = as.numeric(.data[[ref_col]])
        ) %>%
        # Drop NA only (zeros are meaningful)
        filter(
            !is.na(Program),
            !is.na(Reference)
        )
    
    # ---- CRITICAL CHECK ----
    if (nrow(df) < 3) {
        cat(
            "\n--- DEMING REGRESSION SKIPPED ---\n",
            "Reason: At least 3 paired observations are required.\n",
            "After filtering, only ", nrow(df), " valid rows remain.\n",
            sep = ""
        )
        
        return(invisible(NULL))
    }
    
    # ---- Deming regression ----
    deming_fit <- mcreg(
        x = df$Reference, 
        y = df$Program,
        method.reg = "Deming",
        method.ci = "bootstrap",
        error.ratio = 1,
        alpha = 0.05
    )
    
    coefs <- mcr::getCoefficients(deming_fit)
    est_col <- if ("EST" %in% colnames(coefs)) "EST" else "Estimate"
    
    intercept <- coefs["Intercept", est_col]
    slope     <- coefs["Slope", est_col]
    
    p <- ggplot(df, aes(x = Reference, y = Program)) +
        geom_point(alpha = 0.6, color = "#2c3e50") +
        geom_abline(intercept = 0, slope = 1,
                    linetype = "dashed", color = "#7f8c8d", linewidth = 1) +
        geom_abline(intercept = intercept, slope = slope,
                    color = "#c0392b", linewidth = 1.2) +
        labs(
            title = paste("Deming Regression:", title),
            subtitle = paste0(
                "Slope: ", round(slope, 3),
                " | Intercept: ", round(intercept, 2),
                " | N = ", nrow(df)
            ),
            x = "Human Reference Count (Colonies)",
            y = "Program Count (Colonies)"
        ) +
        theme_minimal() +
        theme(plot.title = element_text(face = "bold", hjust = 0.5))
    
    print(p)
    
    correction_formula <- paste0(
        "Corrected_C = (C - ", round(intercept, 2),
        ") / ", round(slope, 3)
    )
    
    list(
        slope = slope,
        intercept = intercept,
        correction_formula = correction_formula,
        N = nrow(df)
    )
}

# --- 3. CCC FUNCTION ---

analyze_ccc <- function(data, program_col, ref_col, title) {
    
    df <- data %>%
        # Filter out zero counts, although CCC can handle them, it ensures comparison consistency
        filter(.data[[program_col]] > 0 | .data[[ref_col]] > 0) %>%
        mutate(
            Program = .data[[program_col]],
            Reference = .data[[ref_col]]
        )
    
    ccc_val <- NA
    ccc_ci_lower <- NA
    ccc_ci_upper <- NA
    
    # CCC is available in the 'DescTools' package (recommended)
    if (requireNamespace("DescTools", quietly = TRUE)) {
        # Ensure data is numeric and vectors are provided
        ccc_result <- DescTools::CCC(as.numeric(df$Reference), as.numeric(df$Program))
        ccc_val <- ccc_result$rho.c$est
        ccc_ci_lower <- ccc_result$rho.c$lwr.ci
        ccc_ci_upper <- ccc_result$rho.c$upr.ci
    } else {
        # Manual Calculation (Fall-back)
        cov_val <- cov(df$Reference, df$Program)
        mean_ref <- mean(df$Reference)
        mean_prog <- mean(df$Program)
        var_ref <- var(df$Reference)
        var_prog <- var(df$Program)
        
        ccc_val <- (2 * cov_val) / (var_ref + var_prog + (mean_ref - mean_prog)^2)
    }
    
    cat("\n--- CONCORDANCE CORRELATION COEFFICIENT (CCC) RESULTS ---\n")
    cat(paste0("Comparison: ", title, "\n"))
    cat(paste0("CCC Value: ", round(ccc_val, 4), "\n"))
    
    if (!is.na(ccc_ci_lower)) {
        cat(paste0("95% CI: [", round(ccc_ci_lower, 4), " to ", round(ccc_ci_upper, 4), "]\n"))
    } else {
        cat("Note: CI not available. Install 'DescTools' for confidence intervals.\n")
    }
    
    # Return metrics
    list(
        ccc = ccc_val,
        ci_lower = ccc_ci_lower,
        ci_upper = ccc_ci_upper
    )
}

# --- 4. CCC MATRIX PLOT FUNCTION ---

# Function to calculate CCC for all pairs and plot a heatmap
plot_ccc_matrix <- function(data, cols_to_analyze) {
    
    if (!requireNamespace("DescTools", quietly = TRUE)) {
        cat("\n--- CCC MATRIX FAILED ---\n")
        cat("Error: 'DescTools' package is required for CCC. Please install it with install.packages(\"DescTools\").\n")
        return(invisible(NULL))
    }
    
    df_ccc <- data %>% 
        select(all_of(cols_to_analyze))
    
    n_cols <- length(cols_to_analyze)
    
    ccc_matrix <- matrix(
        NA, nrow = n_cols, ncol = n_cols,
        dimnames = list(cols_to_analyze, cols_to_analyze)
    )
    
    for (i in 1:n_cols) {
        for (j in 1:n_cols) {
            
            if (i == j) {
                ccc_matrix[i, j] <- 1.0
                
            } else if (i < j) {
                
                x <- df_ccc[[cols_to_analyze[i]]]
                y <- df_ccc[[cols_to_analyze[j]]]
                
                # ---- PAIRWISE COMPLETE CASES ----
                keep <- !is.na(x) & !is.na(y)
                x_pair <- x[keep]
                y_pair <- y[keep]
                
                if (length(x_pair) < 3) {
                    ccc_value <- NA
                } else {
                    ccc_result <- DescTools::CCC(x_pair, y_pair)
                    ccc_value <- ccc_result$rho.c$est
                }
                
                ccc_matrix[i, j] <- ccc_value
                ccc_matrix[j, i] <- ccc_value
            }
        }
    }
    
    ccc_df <- as.data.frame(ccc_matrix) %>%
        rownames_to_column(var = "Method1") %>%
        pivot_longer(-Method1, names_to = "Method2", values_to = "CCC") %>%
        mutate(
            CCC_Label = ifelse(
                is.na(CCC), "NA",
                format(round(CCC, 3), nsmall = 3)
            ),
            Method1 = factor(Method1, levels = cols_to_analyze),
            Method2 = factor(Method2, levels = rev(cols_to_analyze))
        )
    
    p <- ggplot(ccc_df, aes(x = Method1, y = Method2, fill = CCC)) +
        geom_tile(color = "white", linewidth = 0.5) +
        geom_text(aes(label = CCC_Label), color = "black", size = 4) +
        scale_fill_gradient2(
            low = "#e74c3c",
            mid = "#27ae60",
            high = "#27ae60",
            midpoint = 0.95,
            limits = c(0, 1),
            na.value = "grey90",
            name = "Concordance\nCoefficient"
        ) +
        labs(
            title = "Concordance Correlation Coefficient (CCC) Matrix",
            subtitle = "Agreement between all pairs of colony counting methods",
            x = "", y = ""
        ) +
        coord_fixed() +
        theme_minimal() +
        theme(
            axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, face = "bold"),
            axis.text.y = element_text(face = "bold"),
            plot.title = element_text(face = "bold")
        )
    
    print(p)
}

# PARSER 

# Create parser
parser <- ArgumentParser(description = "Generate colony validation report")

# Add arguments
parser$add_argument("-i", "--input", type="character", default="sum.csv",
                    help="Path to CSV with colonies count summary [default %(default)s]")
parser$add_argument("-o", "--output", type="character", default="report.html",
                    help="Path to HTML report [default %(default)s]")
parser$add_argument("-r", "--ref", type="character", default="ref.csv",
                    help="Path to CSV with reference data [default %(default)s]")
parser$add_argument("-t", "--template", type="character", default="report_template.Rmd",
                    help="Path to RMarkdown template [default %(default)s]")
parser$add_argument("--title", type="character", default="Colony Report",
                    help="Report title")
parser$add_argument("--mode", type="character", default="Average_Human",
                    help="Type of referance")


# Parse arguments
args <- parser$parse_args()  

input = args$input
output = args$output
ref = args$ref
template = args$template
titel = args$titel
mode = args$mode


# GETING DATA
input=read.csv(input, row.names = 1)
row.names(input) = toupper(row.names(input))
ref=read.csv(ref, row.names = 1)
row.names(ref) = toupper(row.names(ref))
df <- merge(ref,input,by = "row.names",all = TRUE)

print(head(df))


# ---- Pass parameters to RMarkdown ----
rmarkdown::render(
    input = template,
    output_file = file.path(getwd(),"..", output),
    params = list(
        data = df,
	mode = mode,
        report_title = title
    ),
    envir = new.env()  # isolate environment
)



