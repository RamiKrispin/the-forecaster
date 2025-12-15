#' Plot decomposition of time series data
#'
#' This function generates a plot of the decomposition of time series data into trend,
#' seasonal, and irregular components. Supporting either classical or STL decomposition.
#'
#' @param obj A decompose table object from the feasts library
#' @param var The series variable name (e.g., "trend", "seasonal", etc.)
#' @param outliers Logical; should outliers be shown in the plot? Default is FALSE
#' @return A plotly subplot of the decomposition components


plot_decomposition <- function(obj, var, outliers = FALSE) {
    d <- obj
    obj_attr <- attributes(obj)
    intervals <- unlist(obj_attr$interval)
    interval <- names(which(intervals == 1))
    index <- as.character(obj_attr$index)
    if (interval %in% c("week", "month", "quarter")) {
        d$date <- as.Date(d[[index]])
    } else {
        d$date <- d[[index]]
    }

    color <- "#0072B5"



    if (outliers) {
        if (obj_attr$method == "STL") {
            sdv <- sd(d$remainder)

            d$sd3 <- ifelse(d$remainder >= 3 * sdv | d$remainder <= -3 * sdv, d[[var]], NA)
            d$sd2 <- ifelse(d$remainder >= 2 * sdv & d$remainder < 3 * sdv | d$remainder <= -2 * sdv & d$remainder > -3 * sdv, d[[var]], NA)
        } else {
            sdv <- sd(d$random, na.rm = TRUE)

            d$sd3 <- ifelse(d$random >= 3 * sdv | d$random <= -3 * sdv, d[[var]], NA)
            d$ sd2 <- ifelse(d$random >= 2 * sdv & d$random < 3 * sdv | d$random <= -2 * sdv & d$random > -3 * sdv, d[[var]], NA)
        }
    }


    series <- d |>
        plotly::plot_ly(x = ~date, y = ~ get(var), type = "scatter", mode = "lines", line = list(color = color), name = "Actual", showlegend = FALSE) |>
        plotly::layout(yaxis = list(title = "Actial"))


    trend <- d |>
        plotly::plot_ly(x = ~date, y = ~trend, type = "scatter", mode = "lines", line = list(color = color), name = "Trend", showlegend = FALSE) |>
        plotly::layout(yaxis = list(title = "Trend"))


    if (obj_attr$method == "STL") {
        if (interval != "year") {
            seasonal <- d |>
                plotly::plot_ly(x = ~date, y = ~season_year, type = "scatter", mode = "lines", line = list(color = color), name = "Seasonal", showlegend = FALSE) |>
                plotly::layout(yaxis = list(title = "Seasonal"))
        } else {
            seasonal <- NULL
        }

        seasonal_adj <- d |>
            plotly::plot_ly(x = ~date, y = ~season_adjust, type = "scatter", mode = "lines", line = list(color = color), name = "Seasonal Adjusted", showlegend = FALSE) |>
            plotly::layout(yaxis = list(title = "Seasonal Adjusted"))

        irregular <- d |> plotly::plot_ly(
            x = ~date, y = ~remainder,
            type = "scatter", mode = "lines",
            line = list(color = color), name = "Irregular", showlegend = FALSE
        )
    } else {
        if (interval != "year") {
            seasonal <- d |>
                plotly::plot_ly(x = ~date, y = ~seasonal, type = "scatter", mode = "lines", line = list(color = color), name = "Seasonal", showlegend = FALSE) |>
                plotly::layout(yaxis = list(title = "Seasonal"))
        } else {
            seasonal <- NULL
        }

        seasonal_adj <- d |>
            plotly::plot_ly(x = ~date, y = ~season_adjust, type = "scatter", mode = "lines", line = list(color = color), name = "Seasonal Adjusted", showlegend = FALSE) |>
            plotly::layout(yaxis = list(title = "Seasonal Adjusted"))

        irregular <- d |>
            plotly::plot_ly(
                x = ~date, y = ~random,
                type = "scatter", mode = "lines",
                line = list(color = color), name = "Irregular", showlegend = FALSE
            ) |>
            plotly::layout(yaxis = list(title = "Irregular"))
    }


    if (outliers) {
        series <- series |>
            plotly::add_trace(x = ~date, y = ~sd2, marker = list(color = "orange")) |>
            plotly::add_trace(x = ~date, y = ~sd3, marker = list(color = "red"))
        irregular <- irregular |>
            plotly::add_segments(
                x = min(d$date),
                xend = max(d$date),
                y = 2 * sdv,
                yend = 2 * sdv,
                name = "2SD",
                line = list(color = "orange", dash = "dash")
            ) |>
            plotly::add_segments(
                x = min(d$date),
                xend = max(d$date),
                y = -2 * sdv,
                yend = -2 * sdv,
                name = "-2SD",
                line = list(color = "orange", dash = "dash")
            ) |>
            plotly::add_segments(
                x = min(d$date),
                xend = max(d$date),
                y = 3 * sdv,
                yend = 3 * sdv,
                name = "3SD",
                line = list(color = "red", dash = "dash")
            ) |>
            plotly::add_segments(
                x = min(d$date),
                xend = max(d$date),
                y = -3 * sdv,
                yend = -3 * sdv,
                name = "-3SD",
                line = list(color = "red", dash = "dash")
            ) |>
            plotly::layout(yaxis = list(title = "Irregular"))
    }
    if (is.null(seasonal)) {
        p <- plotly::subplot(series, trend, seasonal_adj, irregular,
            nrows = 4, titleY = TRUE, shareX = TRUE
        )
    } else {
        p <- plotly::subplot(series, trend, seasonal, seasonal_adj, irregular,
            nrows = 5, titleY = TRUE, shareX = TRUE
        )
    }
    capitalize_first <- function(word) {
        if (!is.character(word) || length(word) != 1) {
            stop("Input must be a single character string")
        }
        return(paste0(toupper(substr(word, 1, 1)), tolower(substr(word, 2, nchar(word)))))
    }

    p <- p |>
        plotly::layout(xaxis = list(title = paste("Decomposition Method: ",
            obj_attr$method,
            "; Frequency: ",
            capitalize_first(interval),
            sep = ""
        )))

    return(p)
}


#' Compute and plot ACF of a time series
#'
#' @param ts Time series data
#' @param var Variable name to be plotted
#' @param lag_max Maximum number of lags
#' @param frequency Frequency at which seasonal component is expected (default = NULL)
#'
#' @return A plotly object

plot_acf <- function(ts, var, lag_max, frequency, alpha = 0.05) {
    a <- ts |> feasts::ACF(!!rlang::sym(var), lag_max = lag_max)
    color <- "#0072B5"
    pi_upper <- qnorm(1 - alpha / 2) / sqrt(nrow(ts))
    pi_upper
    p <- plotly::plot_ly(type = "bar")

    if (!is.null(frequency)) {
        s <- seq(from = frequency, by = frequency, to = nrow(a))
        a$seasonal <- NA
        a$non_seasonal <- a$acf
        a$non_seasonal[s] <- NA
        a$seasonal[s] <- a$acf[s]

        p <- p |>
            plotly::add_trace(x = a$lag, y = a$non_seasonal, name = "Non-seasonal", marker = list(
                color = color,
                line = list(
                    color = "rgb(8,48,107)",
                    width = 1.5
                )
            )) |>
            plotly::add_trace(x = a$lag, y = a$seasonal, name = "Seasonal", marker = list(color = "red", line = list(
                color = "rgb(8,48,107)",
                width = 1.5
            )))
    } else {
        p <- p |> plotly::add_trace(x = a$lag, y = a$acf, name = "Lags", marker = list(
            color = color,
            line = list(
                color = "rgb(8,48,107)",
                width = 1.5
            )
        ))
    }

    p <- p |>
        plotly::layout("ACF Plot", yaxis = list(title = "ACF"), xaxis = list(title = "Lags")) |>
        plotly::add_segments(x = ~ min(a$lag), xend = ~ max(a$lag), y = pi_upper, yend = pi_upper, line = list(color = "black", dash = "dash"), name = "95% CI", showlegend = TRUE, legendgroup = "ci") |>
        plotly::add_segments(x = ~ min(a$lag), xend = ~ max(a$lag), y = -pi_upper, yend = -pi_upper, line = list(color = "black", dash = "dash"), name = "95% CI", showlegend = FALSE, legendgroup = "ci")


    return(p)
}



#' Plot a time series against its lagged value with regression line and metrics
#'
#' @param ts A data frame containing a single time series column.
#' @param var The name of the variable to plot.
#' @param lag The number of lags to consider.
#'
#' @return A `plotly` object showing the relationship between the original
#'         variable and its lagged value, along with a regression line and metrics.
#'

plot_lag <- function(ts, var, lag) {
    d <- ts |>
        dplyr::mutate(lag = dplyr::lag(x = !!rlang::sym(var), n = lag))

    # Create the regression formula
    formula <- as.formula(paste(var, "~ lag"))

    # Fit the linear model
    model <- lm(formula, data = d)

    # Extract model coefficients
    intercept <- coef(model)[1]
    slope <- coef(model)[2]

    # Format regression formula text
    reg_formula <- paste0(
        "y = ", round(intercept, 2),
        ifelse(slope < 0, " - ", " + "),
        abs(round(slope, 2)), paste("*lag", lag, sep = "")
    )

    # Get adjusted R-squared
    adj_r2 <- summary(model)$adj.r.squared
    adj_r2_label <- paste0("Adjusted R² = ", round(adj_r2, 3))

    # Add predicted values to data
    d$predicted <- predict(model, newdata = d)

    # Create plot
    p <- plot_ly(d,
        x = ~lag, y = ~ get(var), type = "scatter", mode = "markers",
        name = "Actual"
    ) %>%
        add_lines(
            x = ~lag, y = ~predicted, name = "Regression Fitted Line",
            line = list(color = "red", dash = "dash")
        ) %>%
        layout(
            title = paste(var, "vs Lag", lag, sep = " "),
            xaxis = list(title = paste("Lag", lag, sep = " ")),
            yaxis = list(title = var),
            annotations = list(
                list(
                    x = 0.05, y = 0.95, xref = "paper", yref = "paper",
                    text = reg_formula,
                    showarrow = FALSE,
                    font = list(size = 12)
                ),
                list(
                    x = 0.05, y = 0.88, xref = "paper", yref = "paper",
                    text = adj_r2_label,
                    showarrow = FALSE,
                    font = list(size = 12)
                )
            )
        )

    return(p)
}


#' Piecewise Linear Regression with Grid Search
#'
#' @param data A data frame or tsibble with time series data
#' @param time_col Name of the time column (as string)
#' @param value_col Name of the value column (as string)
#' @param max_knots Maximum number of knots to test (default: 3)
#' @param min_segment_length Minimum number of observations per segment (default: 30)
#' @param edge_buffer Percentage of data to exclude from edges (default: 0.05)
#' @param grid_resolution Number of candidate positions per knot (default: 20)
#' @param record_search Logical; if TRUE, plots each configuration for camcorder recording (default: FALSE)
#' @param plot_dir Directory to save plots if record_search is TRUE (default: NULL uses tempdir)
#' @param plot_dpi DPI for saving plots (default: 80)
#'
#' @return List with optimal model, knot positions, BIC scores, and fitted values
piecewise_regression <- function(data,
                                 time_col = "date",
                                 value_col = "value",
                                 max_knots = 3,
                                 min_segment_length = 30,
                                 edge_buffer = 0.05,
                                 grid_resolution = 20,
                                 record_search = FALSE,
                                 plot_dir = NULL,
                                 plot_dpi = 80) {
    # Prepare data
    df <- data %>%
        arrange(!!sym(time_col)) %>%
        mutate(
            time_index = 1:n(),
            y = !!sym(value_col)
        )

    n <- nrow(df)

    # Define valid range for knots (exclude edges)
    min_idx <- ceiling(n * edge_buffer)
    max_idx <- floor(n * (1 - edge_buffer))

    # Function to fit piecewise linear model given knot positions
    fit_piecewise <- function(knots, data_df) {
        if (length(knots) == 0) {
            # No knots - simple linear regression
            model <- lm(y ~ time_index, data = data_df)
            return(list(
                model = model,
                rss = sum(residuals(model)^2),
                n_params = 2 # intercept + slope
            ))
        }

        # Sort knots
        knots <- sort(knots)

        # Create piecewise linear model using splines with continuity constraint
        # Build the design matrix manually for continuous piecewise linear
        X <- matrix(1, nrow = n, ncol = 1) # Intercept
        X <- cbind(X, data_df$time_index) # First slope

        # Add broken stick terms (continuous piecewise linear)
        for (k in knots) {
            X <- cbind(X, pmax(data_df$time_index - k, 0))
        }

        # Fit model
        model <- lm(data_df$y ~ X - 1) # -1 removes duplicate intercept

        n_params <- 2 + length(knots) # intercept + initial slope + slope changes

        return(list(
            model = model,
            rss = sum(residuals(model)^2),
            n_params = n_params
        ))
    }

    # Function to calculate BIC
    calc_bic <- function(rss, n, k) {
        n * log(rss / n) + k * log(n)
    }

    # Function to generate candidate knot positions
    generate_candidates <- function(n_knots, min_idx, max_idx, min_segment) {
        if (n_knots == 0) {
            return(list(integer(0)))
        }

        # Calculate minimum spacing required
        total_min_length <- (n_knots + 1) * min_segment
        available_length <- max_idx - min_idx + 1

        if (total_min_length > available_length) {
            warning(paste("Cannot fit", n_knots, "knots with min segment length", min_segment))
            return(list())
        }

        # Grid search approach
        candidates <- list()

        if (n_knots == 1) {
            # For 1 knot, test positions with proper spacing
            positions <- seq(min_idx + min_segment,
                max_idx - min_segment,
                length.out = min(grid_resolution, max_idx - min_idx - 2 * min_segment)
            )
            positions <- round(positions)
            candidates <- as.list(positions)
        } else if (n_knots == 2) {
            # For 2 knots, test grid of positions
            pos1_range <- seq(min_idx + min_segment,
                max_idx - 2 * min_segment,
                length.out = min(grid_resolution, available_length / 3)
            )
            pos1_range <- round(pos1_range)

            for (pos1 in pos1_range) {
                pos2_range <- seq(pos1 + min_segment,
                    max_idx - min_segment,
                    length.out = min(grid_resolution, (max_idx - pos1 - min_segment) / min_segment)
                )
                pos2_range <- round(pos2_range)

                for (pos2 in pos2_range) {
                    candidates[[length(candidates) + 1]] <- c(pos1, pos2)
                }
            }
        } else {
            # For 3+ knots, use a coarser grid
            # Divide the range into n_knots+1 segments and place knots at boundaries
            segment_length <- (max_idx - min_idx) / (n_knots + 1)

            # Create base positions evenly spaced
            base_positions <- round(seq(min_idx + segment_length,
                max_idx - segment_length,
                length.out = n_knots
            ))

            # Test variations around base positions
            search_window <- round(segment_length * 0.3)

            # For simplicity with 3+ knots, test fewer variations
            if (n_knots == 3) {
                for (offset1 in seq(-search_window, search_window, length.out = 5)) {
                    for (offset2 in seq(-search_window, search_window, length.out = 5)) {
                        for (offset3 in seq(-search_window, search_window, length.out = 5)) {
                            pos <- round(base_positions + c(offset1, offset2, offset3))
                            # Check minimum segment constraint
                            if (all(diff(c(min_idx, pos, max_idx)) >= min_segment)) {
                                candidates[[length(candidates) + 1]] <- pos
                            }
                        }
                    }
                }
            } else {
                # For 4+ knots, just use base positions
                candidates <- list(base_positions)
            }
        }

        return(candidates)
    }

    # Test different numbers of knots
    results <- list()

    # Track overall best for animation
    overall_best_bic <- Inf
    config_count <- 0 # Track total configurations tested

    for (k in 0:max_knots) {
        cat("Testing", k, "knot(s)...\n")

        # Generate candidate knot positions
        candidates <- generate_candidates(k, min_idx, max_idx, min_segment_length)

        if (length(candidates) == 0) {
            next
        }

        best_bic <- Inf
        best_knots <- NULL
        best_model <- NULL
        best_rss <- NULL

        # Test each candidate
        for (knots in candidates) {
            fit <- fit_piecewise(knots, df)
            bic <- calc_bic(fit$rss, n, fit$n_params)
            config_count <- config_count + 1

            # Plot if recording is enabled
            if (record_search) {
                # Get fitted values for current configuration
                fitted_vals <- fitted(fit$model)

                # Determine if this is the new best
                is_current_best <- bic < best_bic
                is_overall_best <- bic < overall_best_bic

                # Create the plot
                p <- ggplot2::ggplot(df, ggplot2::aes(x = .data[[time_col]], y = .data$y)) +
                    ggplot2::geom_point(alpha = 0.5, size = 1.5, color = "steelblue") +
                    ggplot2::geom_line(ggplot2::aes(y = fitted_vals),
                        color = ifelse(is_overall_best, "darkgreen",
                            ifelse(is_current_best, "steelblue", "gray50")
                        ),
                        linewidth = 1.2
                    ) +
                    ggplot2::labs(
                        title = sprintf(
                            "Grid Search: Testing %d Knot(s) | Config #%d",
                            k, config_count
                        ),
                        subtitle = ifelse(is_overall_best, "NEW OVERALL BEST!",
                            ifelse(is_current_best, "New best for this knot count", "")
                        ),
                        x = time_col,
                        y = value_col
                    ) +
                    ggplot2::theme_minimal(base_size = 14) +
                    ggplot2::theme(
                        plot.title = ggplot2::element_text(face = "bold", size = 16),
                        plot.subtitle = ggplot2::element_text(
                            color = ifelse(is_overall_best, "darkgreen", "steelblue"),
                            face = "bold",
                            size = 13
                        ),
                        plot.background = ggplot2::element_rect(fill = "white", color = NA),
                        panel.background = ggplot2::element_rect(fill = "white", color = NA)
                    )

                # Add vertical lines for knots
                if (length(knots) > 0) {
                    knot_times <- df[[time_col]][knots]
                    for (kt in knot_times) {
                        p <- p + ggplot2::geom_vline(
                            xintercept = kt,
                            linetype = "dashed",
                            color = "red",
                            linewidth = 0.8,
                            alpha = 0.7
                        )
                    }
                }

                # Add BIC score annotation
                p <- p + ggplot2::annotate("text",
                    x = -Inf, y = Inf,
                    label = sprintf(
                        "BIC: %.2f\nKnots: %d\nRSS: %.2f",
                        bic, k, fit$rss
                    ),
                    hjust = -0.1, vjust = 1.2,
                    size = 5,
                    fontface = "bold",
                    color = ifelse(is_overall_best, "darkgreen", "gray20")
                )

                # Save plot manually to file
                if (!is.null(plot_dir)) {
                    frame_file <- file.path(plot_dir, sprintf("frame_%04d.png", config_count))
                    ggplot2::ggsave(
                        filename = frame_file,
                        plot = p,
                        width = 8,
                        height = 6,
                        dpi = plot_dpi,
                        device = "png"
                    )
                    cat("  Saved frame", config_count, "to", basename(frame_file), "\n")
                } else {
                    # Just print if no directory specified
                    print(p)
                }
            }

            if (bic < best_bic) {
                best_bic <- bic
                best_knots <- knots
                best_model <- fit$model
                best_rss <- fit$rss
            }

            # Update overall best
            if (bic < overall_best_bic) {
                overall_best_bic <- bic
            }
        }

        results[[k + 1]] <- list(
            n_knots = k,
            knots = best_knots,
            bic = best_bic,
            rss = best_rss,
            model = best_model,
            n_candidates = length(candidates)
        )

        cat(
            "  Best BIC:", round(best_bic, 2), "| RSS:", round(best_rss, 2),
            "| Tested", length(candidates), "configurations\n"
        )
    }

    # Find optimal number of knots
    bic_values <- sapply(results, function(x) x$bic)
    optimal_idx <- which.min(bic_values)
    optimal <- results[[optimal_idx]]

    cat("\nOptimal model: ", optimal$n_knots, "knot(s) with BIC =", round(optimal$bic, 2), "\n")

    # Get fitted values for optimal model
    df$fitted <- fitted(optimal$model)

    # Convert knot indices back to original time values
    if (length(optimal$knots) > 0) {
        knot_dates <- df[[time_col]][optimal$knots]
    } else {
        knot_dates <- NULL
    }

    return(list(
        optimal_knots = optimal$n_knots,
        knot_positions = optimal$knots,
        knot_dates = knot_dates,
        bic_scores = data.frame(
            n_knots = sapply(results, function(x) x$n_knots),
            bic = bic_values,
            rss = sapply(results, function(x) x$rss)
        ),
        model = optimal$model,
        data = df,
        all_results = results
    ))
}


#' Plot BIC Scores by Number of Knots using Plotly
#'
#' This function creates an interactive plotly visualization of BIC scores
#' across different numbers of knots, highlighting the optimal choice.
#'
#' @param result Output from the piecewise_regression function
#'
#' @return A plotly object showing BIC scores by number of knots
#'
plot_bic_scores <- function(result) {
    # Extract BIC scores data frame
    bic_data <- result$bic_scores
    optimal_knots <- result$optimal_knots

    # Get the optimal point data
    optimal_point <- bic_data[bic_data$n_knots == optimal_knots, ]

    # Create the plot
    p <- plotly::plot_ly() |>
        # Add line
        plotly::add_trace(
            data = bic_data,
            x = ~n_knots,
            y = ~bic,
            type = "scatter",
            mode = "lines+markers",
            line = list(color = "steelblue", width = 2),
            marker = list(size = 8, color = "steelblue"),
            name = "BIC Score",
            showlegend = FALSE
        ) |>
        # Add optimal point highlight
        plotly::add_trace(
            data = optimal_point,
            x = ~n_knots,
            y = ~bic,
            type = "scatter",
            mode = "markers",
            marker = list(size = 12, color = "red"),
            name = "Optimal",
            showlegend = TRUE
        ) |>
        # Add annotation for optimal point
        plotly::layout(
            title = list(
                text = "BIC Scores by Number of Knots<br><sub>Lower BIC = Better Model</sub>",
                font = list(size = 16)
            ),
            xaxis = list(
                title = "Number of Knots",
                dtick = 1
            ),
            yaxis = list(
                title = "BIC"
            ),
            annotations = list(
                list(
                    x = optimal_point$n_knots,
                    y = optimal_point$bic,
                    text = "Optimal",
                    xanchor = "left",
                    xshift = 10,
                    showarrow = TRUE,
                    arrowhead = 2,
                    arrowsize = 1,
                    arrowwidth = 2,
                    arrowcolor = "red",
                    ax = 40,
                    ay = 0,
                    font = list(
                        size = 12,
                        color = "red",
                        family = "Arial Black"
                    )
                )
            ),
            hovermode = "closest"
        )

    return(p)
}


#' Plot Time Series with Optimal Knot Positions
#'
#' This function creates a plotly visualization of the time series data
#' with vertical dashed lines indicating the optimal knot positions
#' found by piecewise regression.
#'
#' @param result Output from the piecewise_regression function
#' @param time_col Name of the time column (as string)
#' @param value_col Name of the value column (as string)
#'
#' @return A plotly object showing the time series with knot positions
#'
#' @examples
#' \dontrun{
#' pw <- piecewise_regression(data = ts1, time_col = "index", value_col = "y")
#' plot_knots(pw, time_col = "index", value_col = "y")
#' }
plot_knots <- function(result, time_col, value_col) {
    # Extract data
    data <- result$data
    knot_dates <- result$knot_dates
    optimal_knots <- result$optimal_knots

    # Create base plot with actual values
    p <- plotly::plot_ly() |>
        plotly::add_trace(
            x = data[[time_col]],
            y = data[[value_col]],
            type = "scatter",
            mode = "markers",
            marker = list(color = "#0072B5", size = 6, opacity = 0.5),
            name = "Actual",
            showlegend = TRUE
        ) |>
        plotly::add_trace(
            x = data[[time_col]],
            y = data$fitted,
            type = "scatter",
            mode = "lines",
            line = list(color = "red", width = 2),
            name = "Fitted (Piecewise)",
            showlegend = TRUE
        )

    # Add vertical lines for each knot position
    if (!is.null(knot_dates) && length(knot_dates) > 0) {
        for (i in seq_along(knot_dates)) {
            p <- p |>
                plotly::add_segments(
                    x = knot_dates[i],
                    xend = knot_dates[i],
                    y = min(data[[value_col]], na.rm = TRUE),
                    yend = max(data[[value_col]], na.rm = TRUE),
                    line = list(color = "darkgreen", dash = "dash", width = 2),
                    name = if (i == 1) "Knots" else NULL,
                    showlegend = if (i == 1) TRUE else FALSE,
                    legendgroup = "knots"
                )
        }
    }

    # Set layout
    p <- p |>
        plotly::layout(
            title = list(
                text = sprintf(
                    "Piecewise Linear Regression<br><sub>Optimal: %d knot%s</sub>",
                    optimal_knots,
                    if (optimal_knots != 1) "s" else ""
                ),
                font = list(size = 16)
            ),
            xaxis = list(title = time_col),
            yaxis = list(title = value_col),
            hovermode = "x unified",
            legend = list(orientation = "h", y = -0.2)
        )

    return(p)
}


#' Record Piecewise Regression Grid Search Animation
#'
#' This function is a wrapper around piecewise_regression that handles
#' the camcorder recording setup and creates an animated GIF of the grid search process.
#'
#' @param data A data frame or tsibble with time series data
#' @param time_col Name of the time column (as string)
#' @param value_col Name of the value column (as string)
#' @param max_knots Maximum number of knots to test (default: 3)
#' @param min_segment_length Minimum number of observations per segment (default: 30)
#' @param edge_buffer Percentage of data to exclude from edges (default: 0.05)
#' @param grid_resolution Number of candidate positions per knot (default: 20)
#' @param output_dir Directory to save animation frames and GIF (default: "grid_search_animation")
#' @param gif_name Name of output GIF file (default: "grid_search.gif")
#' @param width Plot width in pixels (default: 800)
#' @param height Plot height in pixels (default: 600)
#' @param fps Frames per second for GIF (default: 2)
#' @param max_frames Maximum number of frames in GIF, will subsample if exceeded (default: 100)
#' @param dpi DPI for saving PNG frames, higher = better quality but larger files (default: 80)
#' @param gif_width Width in pixels for GIF frames, smaller = faster processing (default: 400)
#'
#' @return List with piecewise regression results and path to GIF
#'
#' @examples
#' \dontrun{
#' # Record the grid search animation
#' result <- record_piecewise_search(
#'     data = ts1,
#'     time_col = "index",
#'     value_col = "y",
#'     max_knots = 2,
#'     output_dir = "animation_frames"
#' )
#'
#' # View the result
#' browseURL(result$gif_path)
#' }
#'
record_piecewise_search <- function(data,
                                    time_col = "date",
                                    value_col = "value",
                                    max_knots = 3,
                                    min_segment_length = 30,
                                    edge_buffer = 0.05,
                                    grid_resolution = 20,
                                    output_dir = "grid_search_animation",
                                    gif_name = "grid_search.gif",
                                    width = 800,
                                    height = 600,
                                    fps = 2,
                                    max_frames = 100,
                                    dpi = 80,
                                    gif_width = 400) {
    # Check if magick is installed
    if (!requireNamespace("magick", quietly = TRUE)) {
        stop("Package 'magick' is required. Install it with: install.packages('magick')")
    }

    # Check data size and provide warnings
    n <- nrow(data)
    min_idx <- ceiling(n * edge_buffer)
    max_idx <- floor(n * (1 - edge_buffer))
    available_length <- max_idx - min_idx + 1

    cat("\n=== Data Analysis ===\n")
    cat("Total observations:", n, "\n")
    cat(
        "Available range for knots:", min_idx, "to", max_idx,
        "(", available_length, "observations )\n"
    )
    cat("Min segment length:", min_segment_length, "\n")
    cat("Max segments needed for", max_knots, "knots:", max_knots + 1, "\n")
    cat("Min observations required:", (max_knots + 1) * min_segment_length, "\n")

    if ((max_knots + 1) * min_segment_length > available_length) {
        recommended_min_seg <- floor(available_length / (max_knots + 1))
        cat("\n⚠️  WARNING: Your dataset may be too small!\n")
        cat("Recommended min_segment_length:", max(3, recommended_min_seg), "\n")
        cat("Or reduce max_knots to:", floor(available_length / min_segment_length) - 1, "\n\n")
    }

    # Create output directory if it doesn't exist
    if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
    }

    # Clean out any old PNG files from previous runs
    old_pngs <- list.files(output_dir, pattern = "^frame_.*\\.png$", full.names = TRUE)
    if (length(old_pngs) > 0) {
        cat("Removing", length(old_pngs), "old PNG files from previous run...\n")
        file.remove(old_pngs)
    }

    # Run piecewise regression with recording enabled
    cat("\n=== Running grid search ===\n")
    cat("PNG quality: ", dpi, " DPI\n")
    result <- piecewise_regression(
        data = data,
        time_col = time_col,
        value_col = value_col,
        max_knots = max_knots,
        min_segment_length = min_segment_length,
        edge_buffer = edge_buffer,
        grid_resolution = grid_resolution,
        record_search = TRUE,
        plot_dir = output_dir,
        plot_dpi = dpi # Pass DPI parameter
    )

    # Check if any frames were captured
    png_files <- list.files(output_dir, pattern = "^frame_.*\\.png$", full.names = TRUE)
    png_files <- sort(png_files) # Ensure correct order

    cat("\n=== Checking captured frames ===\n")
    cat("Output directory:", normalizePath(output_dir), "\n")
    cat("PNG files found:", length(png_files), "\n")

    if (length(png_files) == 0) {
        warning(
            "No animation frames were captured.\n",
            "  Current settings: min_segment_length = ", min_segment_length,
            ", max_knots = ", max_knots, "\n",
            "  Recommended: min_segment_length = ", max(3, floor(available_length / (max_knots + 1)))
        )
        return(list(
            regression_result = result,
            gif_path = NULL,
            frames_dir = output_dir,
            frames_captured = 0
        ))
    }

    # Note: Using all frames with user-specified width
    # Warning about potential issues with large frame counts
    n_frames_original <- length(png_files)

    if (n_frames_original > 150 && gif_width > 400) {
        cat("\n⚠️  Warning: Large frame count (", n_frames_original, " frames) with large width (", gif_width, "px)\n")
        cat("This may cause ImageMagick to crash. Consider using smaller gif_width if issues occur.\n\n")
    }

    # Create GIF using magick
    cat("\n=== Creating GIF with magick ===\n")

    # Check memory availability
    cat("\n=== Memory Check ===\n")
    mem_info <- gc(reset = TRUE)
    cat("R memory before GIF creation:\n")
    print(mem_info)

    gif_path <- file.path(output_dir, gif_name)

    # Remove old GIF if it exists
    if (file.exists(gif_path)) {
        file.remove(gif_path)
    }

    # For very large frame counts, use batch processing
    n_frames_total <- length(png_files)
    batch_size <- 20 # Process 20 frames at a time (reduced from 50)
    use_batch_mode <- n_frames_total > batch_size

    if (use_batch_mode) {
        cat("\n⚡ Using batch mode for", n_frames_total, "frames (processing", batch_size, "at a time)\n")
    }

    tryCatch(
        {
            if (use_batch_mode) {
                # One-at-a-time processing mode to avoid ImageMagick cache exhaustion
                cat("Reading and processing frames one at a time (to avoid cache exhaustion)...\n")

                frames <- NULL

                for (i in 1:n_frames_total) {
                    if (i %% 10 == 1) {
                        cat("  Processing frame", i, "of", n_frames_total, "...\n")
                    }

                    # Read ONE frame at a time
                    single_frame <- magick::image_read(png_files[i])

                    # Resize IMMEDIATELY to minimize cache usage
                    single_frame <- magick::image_scale(single_frame, paste0(gif_width, "x"))

                    # Append to frames collection
                    if (is.null(frames)) {
                        frames <- single_frame
                    } else {
                        frames <- c(frames, single_frame)
                    }

                    # Clear the single frame
                    rm(single_frame)

                    # Garbage collect every 10 frames
                    if (i %% 10 == 0) {
                        invisible(gc(full = TRUE))
                    }
                }

                # Final cleanup
                invisible(gc(full = TRUE))
            } else {
                # Standard mode for fewer frames
                cat("Reading", length(png_files), "frames...\n")
                frames <- magick::image_read(png_files)

                # Resize frames to reduce memory usage
                cat("Resizing frames to", gif_width, "px width...\n")
                frames <- magick::image_scale(frames, paste0(gif_width, "x"))
            }

            # Get number of frames
            n_frames <- length(frames)
            cat("Total frames ready:", n_frames, "\n")

            # Calculate delay in 1/100ths of a second
            delay_normal <- round(100 / fps) # Delay for normal frames
            delay_first <- 300 # 3 seconds for first frame
            delay_last <- 500 # 5 seconds for last frame

            # Build animation
            cat("Building animation with delays...\n")

            # Create delay vector
            delays <- rep(delay_normal, n_frames)
            delays[1] <- delay_first
            if (n_frames > 1) {
                delays[n_frames] <- delay_last
            }

            # Animate with variable delays
            # Disable optimization for large frame counts to prevent segfaults
            use_optimize <- n_frames <= 60
            if (!use_optimize) {
                cat("⚠️  Disabling GIF optimization due to large frame count (", n_frames, " frames)\n")
            }
            cat("Applying animation (this may take a while)...\n")
            animated <- magick::image_animate(frames, delay = delays / 100, optimize = use_optimize)

            # Clear frames from memory immediately
            rm(frames, delays)
            invisible(gc(full = TRUE))

            # Write GIF
            cat("Writing GIF to disk (", n_frames, " frames at ", gif_width, "px)...\n", sep = "")
            cat("This may take a moment...\n")
            magick::image_write(animated, path = gif_path, format = "gif")

            # Clear animated from memory
            rm(animated)
            invisible(gc(full = TRUE))

            # Verify the file was created and has content
            if (file.exists(gif_path) && file.size(gif_path) > 0) {
                cat("\n✅ Animation saved to:", gif_path, "\n")
                cat("📊 Frames in GIF:", n_frames, "\n")
                cat("📦 GIF size:", round(file.size(gif_path) / 1024^2, 2), "MB\n")

                # Show final memory usage
                cat("\n=== Memory after GIF creation ===\n")
                print(gc())
                cat("\n")
            } else {
                stop("GIF file is empty or was not created")
            }
        },
        error = function(e) {
            cat("\n⚠️  Error creating GIF:", e$message, "\n")
            cat("Trying fallback method with smaller frames and batch processing...\n\n")

            tryCatch(
                {
                    # Simpler fallback: smaller size, smaller batches
                    fallback_width <- max(200, round(gif_width * 0.75))
                    cat("Using fallback width:", fallback_width, "px\n")

                    # Process one frame at a time for fallback
                    cat("Processing", length(png_files), "frames one at a time (fallback method)...\n")

                    frames <- NULL

                    for (i in 1:length(png_files)) {
                        if (i %% 10 == 1) {
                            cat("  Fallback processing frame", i, "of", length(png_files), "...\n")
                        }

                        # Read ONE frame at a time
                        single_frame <- magick::image_read(png_files[i])

                        # Resize IMMEDIATELY with smaller size
                        single_frame <- magick::image_scale(single_frame, paste0(fallback_width, "x"))

                        # Append to frames collection
                        if (is.null(frames)) {
                            frames <- single_frame
                        } else {
                            frames <- c(frames, single_frame)
                        }

                        # Clear and collect garbage every 5 frames (more aggressive)
                        rm(single_frame)
                        if (i %% 5 == 0) {
                            invisible(gc(full = TRUE))
                        }
                    }

                    # Final cleanup
                    invisible(gc(full = TRUE))

                    # Use simple fps, no optimization to avoid crashes
                    cat("Animating with simple fps (no optimization)...\n")
                    animated <- magick::image_animate(frames, fps = 2, optimize = FALSE)

                    rm(frames)
                    invisible(gc(full = TRUE))

                    cat("Writing fallback GIF...\n")
                    magick::image_write(animated, path = gif_path, format = "gif")

                    rm(animated)
                    invisible(gc(full = TRUE))

                    # Verify
                    if (file.exists(gif_path) && file.size(gif_path) > 0) {
                        cat("\n✅ Animation saved to:", gif_path, "(using fallback method)\n")
                        cat("📊 Frames in GIF:", length(png_files), "\n")
                        cat("📦 GIF size:", round(file.size(gif_path) / 1024^2, 2), "MB\n\n")
                    } else {
                        stop("Fallback also failed to create GIF")
                    }
                },
                error = function(e2) {
                    warning(
                        "Failed to create GIF: ", e2$message, "\n",
                        "PNG frames are available at: ", normalizePath(output_dir), "\n",
                        "You can manually create the GIF using an external tool or ffmpeg."
                    )
                    return(list(
                        regression_result = result,
                        gif_path = NULL,
                        frames_dir = output_dir,
                        frames_captured = length(list.files(output_dir, pattern = "^frame_.*\\.png$"))
                    ))
                }
            )
        }
    )

    return(list(
        regression_result = result,
        gif_path = gif_path,
        frames_dir = output_dir,
        frames_captured = length(png_files)
    ))
}


#' Plot Residuals Diagnostics
#'
#' Creates a comprehensive residual diagnostic plot with actual vs fitted,
#' residuals with SD lines, ACF plot, Q-Q plot, and density plot.
#'
#' @param data A data frame containing the time series data
#' @param index_col Name of the index/time column (as string)
#' @param actual_col Name of the actual values column (as string)
#' @param fitted_col Name of the fitted values column (as string)
#' @param lag_max Maximum number of lags for ACF plot (default: 60)
#' @param frequency Frequency for seasonal ACF highlighting (default: NULL)
#' @param alpha Significance level for confidence intervals (default: 0.05)
#'
#' @return A plotly subplot object with residual diagnostic plots
plot_residuals <- function(data,
                           index_col,
                           actual_col,
                           fitted_col,
                           lag_max = 60,
                           frequency = NULL,
                           alpha = 0.05) {
    # Calculate residuals
    data$residuals <- data[[actual_col]] - data[[fitted_col]]

    # Remove NA values for diagnostic plots
    residuals_clean <- na.omit(data$residuals)

    # Calculate SD
    sd_res <- sd(residuals_clean)

    # Classify fitted values based on residual magnitude for color coding
    data$residual_category <- ifelse(abs(data$residuals) < 2 * sd_res, "normal",
        ifelse(abs(data$residuals) < 3 * sd_res, "medium", "high")
    )

    # 1. Actual vs Fitted plot
    p_actual_fitted <- plot_ly()

    # Add actual values line
    p_actual_fitted <- p_actual_fitted |>
        add_lines(
            x = data[[index_col]], y = data[[actual_col]],
            name = "Actual",
            line = list(color = "#0072B5"),
            showlegend = TRUE
        )

    # Add fitted values as scatter points, color-coded by residual category
    # Normal fitted values (light blue)
    normal_idx <- which(data$residual_category == "normal")
    if (length(normal_idx) > 0) {
        p_actual_fitted <- p_actual_fitted |>
            add_trace(
                x = data[[index_col]][normal_idx],
                y = data[[fitted_col]][normal_idx],
                type = "scatter",
                mode = "markers",
                marker = list(color = "rgba(135, 206, 250, 0.6)", size = 6),
                name = "Fitted",
                showlegend = TRUE
            )
    }

    # Medium outliers (orange)
    medium_idx <- which(data$residual_category == "medium")
    if (length(medium_idx) > 0) {
        p_actual_fitted <- p_actual_fitted |>
            add_trace(
                x = data[[index_col]][medium_idx],
                y = data[[fitted_col]][medium_idx],
                type = "scatter",
                mode = "markers",
                marker = list(color = "orange", size = 6),
                name = "Fitted (2-3 SD)",
                showlegend = TRUE
            )
    }

    # High outliers (red)
    high_idx <- which(data$residual_category == "high")
    if (length(high_idx) > 0) {
        p_actual_fitted <- p_actual_fitted |>
            add_trace(
                x = data[[index_col]][high_idx],
                y = data[[fitted_col]][high_idx],
                type = "scatter",
                mode = "markers",
                marker = list(color = "red", size = 6),
                name = "Fitted (>3 SD)",
                showlegend = TRUE
            )
    }

    p_actual_fitted <- p_actual_fitted |>
        layout(
            yaxis = list(title = "Value"),
            xaxis = list(title = ""),
            showlegend = TRUE,
            legend = list(orientation = "h")
        )

    # 2. Residuals with SD lines
    p_residuals <- plot_ly() |>
        add_trace(
            x = data[[index_col]], y = data$residuals,
            type = "scatter", mode = "markers",
            name = "Residuals",
            marker = list(color = "#0072B5"),
            showlegend = FALSE
        ) |>
        add_segments(
            x = min(data[[index_col]]), xend = max(data[[index_col]]),
            y = 2 * sd_res, yend = 2 * sd_res,
            line = list(color = "orange", dash = "dash", width = 2),
            name = "+2SD",
            showlegend = FALSE
        ) |>
        add_segments(
            x = min(data[[index_col]]), xend = max(data[[index_col]]),
            y = -2 * sd_res, yend = -2 * sd_res,
            line = list(color = "orange", dash = "dash", width = 2),
            name = "-2SD",
            showlegend = FALSE
        ) |>
        add_segments(
            x = min(data[[index_col]]), xend = max(data[[index_col]]),
            y = 3 * sd_res, yend = 3 * sd_res,
            line = list(color = "red", dash = "dash", width = 2),
            name = "+3SD",
            showlegend = FALSE
        ) |>
        add_segments(
            x = min(data[[index_col]]), xend = max(data[[index_col]]),
            y = -3 * sd_res, yend = -3 * sd_res,
            line = list(color = "red", dash = "dash", width = 2),
            name = "-3SD",
            showlegend = FALSE
        ) |>
        layout(
            yaxis = list(title = "Residuals"),
            xaxis = list(title = "")
        )

    # 3. ACF Plot
    acf_result <- acf(residuals_clean, lag.max = lag_max, plot = FALSE)
    acf_data <- data.frame(
        lag = as.numeric(acf_result$lag),
        acf = as.numeric(acf_result$acf)
    )

    # Remove lag 0
    acf_data <- acf_data[acf_data$lag > 0, ]

    # Calculate confidence interval (use length of clean residuals)
    ci <- qnorm(1 - alpha / 2) / sqrt(length(residuals_clean))

    p_acf <- plot_ly(type = "bar")

    # Handle frequency for seasonal/non-seasonal split
    if (!is.null(frequency)) {
        # Identify seasonal lags
        s <- seq(from = frequency, by = frequency, to = nrow(acf_data))
        acf_data$seasonal <- NA
        acf_data$non_seasonal <- acf_data$acf
        acf_data$non_seasonal[s] <- NA
        acf_data$seasonal[s] <- acf_data$acf[s]

        p_acf <- p_acf |>
            add_trace(
                x = acf_data$lag, y = acf_data$non_seasonal,
                name = "Non-seasonal",
                marker = list(
                    color = "#0072B5",
                    line = list(color = "rgb(8,48,107)", width = 1.5)
                ),
                showlegend = FALSE
            ) |>
            add_trace(
                x = acf_data$lag, y = acf_data$seasonal,
                name = "Seasonal",
                marker = list(
                    color = "red",
                    line = list(color = "rgb(8,48,107)", width = 1.5)
                ),
                showlegend = FALSE
            )
    } else {
        p_acf <- p_acf |>
            add_trace(
                x = acf_data$lag, y = acf_data$acf,
                marker = list(
                    color = "#0072B5",
                    line = list(color = "rgb(8,48,107)", width = 1.5)
                ),
                name = "ACF",
                showlegend = FALSE
            )
    }

    p_acf <- p_acf |>
        add_segments(
            x = min(acf_data$lag), xend = max(acf_data$lag),
            y = ci, yend = ci,
            line = list(color = "black", dash = "dash"),
            name = "95% CI",
            showlegend = FALSE
        ) |>
        add_segments(
            x = min(acf_data$lag), xend = max(acf_data$lag),
            y = -ci, yend = -ci,
            line = list(color = "black", dash = "dash"),
            showlegend = FALSE
        ) |>
        layout(
            yaxis = list(title = "ACF"),
            xaxis = list(title = "Lag")
        )

    # 4. Q-Q Plot
    n <- length(residuals_clean)
    theoretical_quantiles <- qnorm(ppoints(n))
    sample_quantiles <- sort(residuals_clean)
    standardized_res <- (sample_quantiles - mean(residuals_clean)) / sd(residuals_clean)

    p_qq <- plot_ly() |>
        add_trace(
            x = theoretical_quantiles, y = standardized_res,
            type = "scatter", mode = "markers",
            marker = list(color = "#0072B5", size = 6, opacity = 0.6),
            name = "Sample",
            showlegend = FALSE
        ) |>
        add_trace(
            x = theoretical_quantiles, y = theoretical_quantiles,
            type = "scatter", mode = "lines",
            line = list(color = "red", dash = "dash", width = 2),
            name = "Normal",
            showlegend = FALSE
        ) |>
        layout(
            yaxis = list(title = "Sample Quantiles"),
            xaxis = list(title = "Theoretical Quantiles")
        )

    # 5. Density Plot
    density_res <- density(residuals_clean)
    mean_res <- mean(residuals_clean)
    sd_res_density <- sd(residuals_clean)
    x_norm <- seq(min(residuals_clean), max(residuals_clean), length.out = 100)
    y_norm <- dnorm(x_norm, mean = mean_res, sd = sd_res_density)

    p_density <- plot_ly() |>
        add_trace(
            x = density_res$x, y = density_res$y,
            type = "scatter", mode = "lines",
            fill = "tozeroy",
            fillcolor = "rgba(0, 114, 181, 0.3)",
            line = list(color = "#0072B5", width = 2),
            name = "Density",
            showlegend = FALSE
        ) |>
        add_trace(
            x = x_norm, y = y_norm,
            type = "scatter", mode = "lines",
            line = list(color = "red", dash = "dash", width = 2),
            name = "Normal",
            showlegend = FALSE
        ) |>
        layout(
            yaxis = list(title = "Density"),
            xaxis = list(title = "Residuals")
        )

    # Create third row with 3 plots side by side
    row3 <- subplot(
        p_acf, p_qq, p_density,
        nrows = 1,
        shareX = FALSE,
        shareY = FALSE,
        titleX = TRUE,
        titleY = TRUE
    )

    # First create subplot for actual vs fitted and residuals with shared x-axis
    rows_1_2 <- subplot(
        p_actual_fitted,
        p_residuals,
        nrows = 2,
        shareX = TRUE,
        shareY = FALSE,
        titleY = TRUE,
        titleX = TRUE
    )

    # Combine all rows
    subplot_result <- subplot(
        rows_1_2,
        row3,
        nrows = 2,
        heights = c(0.5, 0.5),
        shareX = FALSE,
        shareY = FALSE,
        titleY = TRUE,
        titleX = TRUE
    ) |>
        layout(title = "Residual Plots")

    return(subplot_result)
}


#' Simulate Forecast Paths Using Coefficient Uncertainty
#'
#' Creates forecast simulations by randomly drawing coefficients from their
#' estimated distributions and applying them to future data. Supports recursive
#' forecasting for models with lagged variables.
#'
#' @param model An lm model object
#' @param future_data A data frame with future predictor values
#' @param n_sims Number of simulation paths to generate (default: 1000)
#' @param add_residual_error Logical; add random residual error to predictions (default: TRUE)
#' @param lag_col Optional; name of lag column for recursive forecasting (default: NULL)
#' @param seed Optional; random seed for reproducibility (default: NULL)
#'
#' @return A data frame with future_data plus columns sim_1, sim_2, ..., sim_n
#'
#' @examples
#' \dontrun{
#' # Simple model without lags
#' result <- sim_forecast(model, future_data, n_sims = 1000)
#'
#' # AR model with recursive forecasting
#' result <- sim_forecast(model, future_data, n_sims = 1000, lag_col = "lag1")
#' }
sim_forecast <- function(model,
                         future_data,
                         n_sims = 1000,
                         add_residual_error = TRUE,
                         lag_col = NULL,
                         seed = NULL) {
    # Set seed for reproducibility
    if (!is.null(seed)) {
        set.seed(seed)
    }

    # Extract coefficient estimates and standard errors
    coef_summary <- summary(model)$coefficients
    coef_mean <- coef_summary[, "Estimate"]
    coef_se <- coef_summary[, "Std. Error"]

    # Get residual standard error
    sigma <- summary(model)$sigma

    # Initialize result data frame (copy future_data)
    result <- future_data

    # Run simulations
    for (sim in 1:n_sims) {
        # Draw random coefficients from N(mean, sd)
        sim_coefs <- rnorm(length(coef_mean), mean = coef_mean, sd = coef_se)
        names(sim_coefs) <- names(coef_mean)

        # Make a copy of future_data for this simulation (for recursive forecasting)
        sim_data <- future_data

        # Create prediction vector
        predictions <- numeric(nrow(future_data))

        # Make predictions for each time step
        for (i in 1:nrow(future_data)) {
            # Update lag column if needed (recursive forecasting)
            if (!is.null(lag_col) && i > 1) {
                sim_data[[lag_col]][i] <- predictions[i - 1]
            }

            # Create model matrix for this row
            X_row <- model.matrix(delete.response(terms(model)), data = sim_data[i, , drop = FALSE])

            # Calculate prediction: X * beta
            pred_value <- as.numeric(X_row %*% sim_coefs)

            # Add residual error if requested
            if (add_residual_error) {
                pred_value <- pred_value + rnorm(1, mean = 0, sd = sigma)
            }

            predictions[i] <- pred_value
        }

        # Add to result data frame
        result[[paste0("sim_", sim)]] <- predictions
    }

    return(result)
}


#' Plot Forecast Simulations
#'
#' Creates a plot showing historical actual values and simulated forecast paths.
#'
#' @param actual_data A data frame with historical actual values
#' @param sim_data A data frame with simulation results (output from sim_forecast)
#' @param index_col Name of the index/time column (as string)
#' @param actual_col Name of the actual values column (as string)
#' @param show_intervals Logical; show prediction intervals (default: TRUE)
#' @param interval_levels Vector of probability levels for intervals (default: c(0.5, 0.8, 0.95))
#'
#' @return A plotly object
#'
#' @examples
#' \dontrun{
#' sims <- sim_forecast(model, future_data, n_sims = 1000)
#' plot_sim_forecast(
#'     actual_data = ts1, sim_data = sims,
#'     index_col = "index", actual_col = "y"
#' )
#' }
plot_sim_forecast <- function(actual_data,
                              sim_data,
                              index_col,
                              actual_col,
                              show_intervals = TRUE,
                              interval_levels = c(0.5, 0.8, 0.95)) {
    # Identify simulation columns
    sim_cols <- grep("^sim_", names(sim_data), value = TRUE)

    # Create base plot with actual data
    p <- plot_ly() |>
        add_lines(
            x = actual_data[[index_col]],
            y = actual_data[[actual_col]],
            name = "Actual",
            line = list(color = "#0072B5", width = 2),
            showlegend = TRUE
        )

    # Add all simulation paths
    for (sim_col in sim_cols) {
        p <- p |>
            add_lines(
                x = sim_data[[index_col]],
                y = sim_data[[sim_col]],
                name = "Simulations",
                line = list(color = "rgba(7, 164, 181, 0.2)", width = 1),
                showlegend = FALSE,
                hoverinfo = "skip"
            )
    }

    # Add prediction intervals if requested
    if (show_intervals) {
        # Extract simulation matrix
        sim_matrix <- as.matrix(sim_data[, sim_cols])

        # Calculate quantiles for each row (time step)
        interval_colors <- c(
            "rgba(7, 164, 181, 0.1)",
            "rgba(7, 164, 181, 0.15)",
            "rgba(7, 164, 181, 0.2)"
        )

        # Add intervals from widest to narrowest
        for (i in length(interval_levels):1) {
            level <- interval_levels[i]
            alpha_lower <- (1 - level) / 2
            alpha_upper <- 1 - alpha_lower

            lower <- apply(sim_matrix, 1, quantile, probs = alpha_lower)
            upper <- apply(sim_matrix, 1, quantile, probs = alpha_upper)

            p <- p |>
                add_ribbons(
                    x = sim_data[[index_col]],
                    ymin = lower,
                    ymax = upper,
                    name = paste0(level * 100, "% Interval"),
                    fillcolor = interval_colors[i],
                    line = list(color = "transparent"),
                    showlegend = TRUE,
                    hoverinfo = "skip"
                )
        }

        # Add median line
        median_sim <- apply(sim_matrix, 1, median)
        p <- p |>
            add_lines(
                x = sim_data[[index_col]],
                y = median_sim,
                name = "Median Forecast",
                line = list(color = "rgba(7, 164, 181, 1)", width = 2, dash = "dash"),
                showlegend = TRUE
            )
    }

    # Layout
    p <- p |>
        layout(
            title = "Forecast Simulations",
            xaxis = list(title = ""),
            yaxis = list(title = "Value"),
            hovermode = "x unified",
            legend = list(orientation = "h", y = -0.2)
        )

    return(p)
}


#' Generate Forecast from Linear Model
#'
#' Creates forecasts from a fitted linear model, handling recursive forecasting
#' for models with lagged variables.
#'
#' @param model A fitted lm model object
#' @param actual A data frame with historical actual values
#' @param future_data A data frame with future predictor values
#' @param actual_col Name of the actual values column (as string)
#' @param lag_col Optional vector of lag column names for recursive forecasting (default: NULL)
#' @param level Confidence/prediction interval level (default: 0.95)
#'
#' @return A list with two components:
#'   - actual: The actual data frame with added "fitted" column
#'   - forecast: The future data frame with "yhat", "lower", "upper" columns
#'
#' @examples
#' \dontrun{
#' # Simple model without lags
#' result <- lm_forecast(
#'     model = md1, actual = ts1, future_data = future_data,
#'     actual_col = "y"
#' )
#'
#' # AR model with recursive forecasting
#' result <- lm_forecast(
#'     model = md3, actual = ts1, future_data = future_data,
#'     actual_col = "y", lag_col = "lag1"
#' )
#'
#' # Multiple lags
#' result <- lm_forecast(
#'     model = md4, actual = ts1, future_data = future_data,
#'     actual_col = "y", lag_col = c("lag1", "lag2")
#' )
#' }
lm_forecast <- function(model,
                        actual,
                        future_data,
                        actual_col,
                        lag_col = NULL,
                        level = 0.95) {
    # Fit on actual data (confidence intervals)
    fit_result <- predict(
        object = model,
        newdata = actual,
        interval = "confidence",
        level = level
    )
    actual$fitted <- fit_result[, 1]

    # Handle forecasting
    h <- nrow(future_data)

    if (is.null(lag_col)) {
        # No lags - straightforward prediction
        fc_result <- predict(
            object = model,
            newdata = future_data,
            interval = "prediction",
            level = level
        )

        future_data$yhat <- fc_result[, 1]
        future_data$lower <- fc_result[, 2]
        future_data$upper <- fc_result[, 3]
    } else {
        # Recursive forecasting with lags
        # Extract lag indices from column names
        lag_indices <- as.numeric(gsub("lag", "", lag_col))
        max_lag <- max(lag_indices)

        # Initialize lag columns in future_data with last observations from actual
        n_actual <- nrow(actual)
        for (j in seq_along(lag_col)) {
            lag_idx <- lag_indices[j]
            # For first row: lag1 = actual[n], lag2 = actual[n-1], etc.
            future_data[[lag_col[j]]][1] <- actual[[actual_col]][n_actual - lag_idx + 1]
        }

        # Initialize forecast columns
        future_data$yhat <- NA
        future_data$lower <- NA
        future_data$upper <- NA

        # Forecast each row recursively
        for (i in 1:h) {
            # Update lag columns for current row (if not first row)
            if (i > 1) {
                for (j in seq_along(lag_col)) {
                    lag_idx <- lag_indices[j]
                    if (i > lag_idx) {
                        # Use previous forecast
                        future_data[[lag_col[j]]][i] <- future_data$yhat[i - lag_idx]
                    } else {
                        # Use actual data
                        future_data[[lag_col[j]]][i] <- actual[[actual_col]][n_actual - lag_idx + i]
                    }
                }
            }

            # Forecast current row
            fc_temp <- predict(
                object = model,
                newdata = future_data[i, , drop = FALSE],
                interval = "prediction",
                level = level
            )

            future_data$yhat[i] <- fc_temp[, 1]
            future_data$lower[i] <- fc_temp[, 2]
            future_data$upper[i] <- fc_temp[, 3]
        }
    }

    # Return list
    return(list(
        actual = actual,
        forecast = future_data
    ))
}


#' Plot Linear Model Forecast
#'
#' Creates a plot showing historical actual values, fitted values, and forecasts
#' with prediction intervals.
#'
#' @param result Output from the lm_forecast function (a list with 'actual' and 'forecast' components)
#' @param actual_col Name of the actual values column (as string)
#'
#' @return A plotly object
#'
#' @examples
#' \dontrun{
#' fc <- lm_forecast(
#'     model = md2, actual = ts1, future_data = future_data,
#'     actual_col = "y"
#' )
#' plot_lm_forecast(fc, actual_col = "y")
#' }
plot_lm_forecast <- function(result, actual_col) {
    # Extract index column name from tsibble attributes
    index_col <- as.character(attributes(result$actual)$index)

    # Get data
    actual_data <- result$actual
    forecast_data <- result$forecast

    # Calculate residuals for color-coding fitted values
    actual_data$residuals <- actual_data[[actual_col]] - actual_data$fitted
    sd_res <- sd(actual_data$residuals, na.rm = TRUE)

    # Classify fitted values based on residual magnitude
    actual_data$residual_category <- ifelse(abs(actual_data$residuals) < 2 * sd_res, "normal",
        ifelse(abs(actual_data$residuals) < 3 * sd_res, "medium", "high")
    )

    # Create base plot with actual values
    p <- plot_ly() |>
        add_lines(
            x = actual_data[[index_col]],
            y = actual_data[[actual_col]],
            name = "Actual",
            line = list(color = "#1f77b4"),
            showlegend = TRUE
        )

    # Add fitted values with color-coding (using only normal color)
    normal_idx <- which(actual_data$residual_category == "normal")
    if (length(normal_idx) > 0) {
        p <- p |>
            add_trace(
                x = actual_data[[index_col]][normal_idx],
                y = actual_data$fitted[normal_idx],
                type = "scatter",
                mode = "markers",
                marker = list(color = "rgba(135, 206, 250, 0.6)", size = 6),
                name = "Fitted",
                showlegend = TRUE
            )
    }

    # Add medium outliers (if any)
    medium_idx <- which(actual_data$residual_category == "medium")
    if (length(medium_idx) > 0) {
        p <- p |>
            add_trace(
                x = actual_data[[index_col]][medium_idx],
                y = actual_data$fitted[medium_idx],
                type = "scatter",
                mode = "markers",
                marker = list(color = "rgba(135, 206, 250, 0.6)", size = 6),
                showlegend = FALSE
            )
    }

    # Add high outliers (if any)
    high_idx <- which(actual_data$residual_category == "high")
    if (length(high_idx) > 0) {
        p <- p |>
            add_trace(
                x = actual_data[[index_col]][high_idx],
                y = actual_data$fitted[high_idx],
                type = "scatter",
                mode = "markers",
                marker = list(color = "rgba(135, 206, 250, 0.6)", size = 6),
                showlegend = FALSE
            )
    }

    # Add prediction interval ribbon
    p <- p |>
        add_ribbons(
            x = forecast_data[[index_col]],
            ymin = forecast_data$lower,
            ymax = forecast_data$upper,
            name = "95% PI",
            line = list(color = "rgba(7, 164, 181, 0.05)"),
            fillcolor = "rgba(7, 164, 181, 0.2)",
            showlegend = TRUE
        )

    # Add forecast line
    p <- p |>
        add_lines(
            x = forecast_data[[index_col]],
            y = forecast_data$yhat,
            name = "Forecast",
            line = list(color = "black", dash = "dash"),
            showlegend = TRUE
        )

    # Set layout
    p <- p |>
        layout(
            yaxis = list(title = "Value"),
            xaxis = list(title = ""),
            legend = list(orientation = "h", xanchor = "center", x = 0.5, y = -0.2)
        )

    return(p)
}
