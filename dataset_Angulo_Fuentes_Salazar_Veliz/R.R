
install.packages("readxl")
install.packages("Metrics")
install.packages("rpart")
install.packages("rpart.plot")

bd <- social_media_vs_productivity
head(bd)

#IMPUTACIÓN DE VALORES FALTANTES (NA)
bd[] <- lapply(bd, function(x) {
  if (is.numeric(x)) {
    x[is.na(x)] <- mean(x, na.rm = TRUE)
  }
  return(x)
})

# ANÁLISIS EXPLORATORIO: CORRELACIONES
vars_numericas <- c("actual_productivity_score", "perceived_productivity_score",
                    "job_satisfaction_score", "stress_level", "sleep_hours",
                    "work_hours_per_day", "number_of_notifications",
                    "days_feeling_burnout_per_month", "weekly_offline_hours",
                    "screen_time_before_sleep", "breaks_during_work")
#Matriz de correlación
cor(bd[, vars_numericas], use = "complete.obs")

#CODIFICACIÓN DE VARIABLE CATEGÓRICA
# Creamos dummy binaria: 1 si es desempleado, 0 si tiene empleo
bd$es_desempleado <- ifelse(bd$job_type == "Unemployed", 1, 0)

#SPLIT TRAIN / TEST
set.seed(123)
idx   <- sample(1:nrow(bd), size = 0.8 * nrow(bd))
train <- bd[idx, ]
test  <- bd[-idx, ]

#MODELO 1: REGRESIÓN LINEAL MÚLTIPLE (MLR)
modelo_mlr <- lm(actual_productivity_score ~ job_satisfaction_score +
                   es_desempleado, data = train)

# Resumen completo: coeficientes, error estándar, p-values, R² y F-statistic
summary(modelo_mlr)

# Intervalos de confianza al 95% para cada coeficiente
confint(modelo_mlr, level = 0.95)

# Predicciones del MLR sobre el conjunto de prueba
library(Metrics)
pred_mlr <- predict(modelo_mlr, newdata = test)

cat("── Métricas MLR en test ──\n")
rmse_mlr <- rmse(test$actual_productivity_score, pred_mlr)
mae_mlr  <- mae(test$actual_productivity_score,  pred_mlr)
cat("RMSE MLR:      ", rmse_mlr, "\n")
cat("MAE  MLR:      ", mae_mlr,  "\n")

#VERIFICACIÓN DE SUPUESTOS DEL MLR
# Muestra de 500 filas: los gráficos con 24.000 puntos son ilegibles
datos_pequeños <- train[sample(nrow(train), 500), ]

# Modelo sobre la muestra solo para gráficos diagnósticos, no para métricas
modelo_muestra <- lm(actual_productivity_score ~ job_satisfaction_score +
                       es_desempleado, data = datos_pequeños)
par(mfrow = c(2, 2))
plot(modelo_muestra)
par(mfrow = c(1, 1))

#MODELO 2: ÁRBOL DE DECISIÓN

library(rpart)
library(rpart.plot)

modelo_arbol <- rpart(
  actual_productivity_score ~ job_satisfaction_score + es_desempleado,
  data   = train,
  method = "anova",
  control = rpart.control(cp = 0.001)
)

rpart.plot(modelo_arbol, type = 4, extra = 101,
           main = "Árbol de Decisión — actual_productivity_score")

# Muestra la tabla de complejidad:
printcp(modelo_arbol)

# Selecciona el valor de cp que minimiza el error de validación cruzada y poda el árbol para evitar sobreajuste
cp_optimo <- modelo_arbol$cptable[which.min(modelo_arbol$cptable[, "xerror"]), "CP"]
cat("CP óptimo:", cp_optimo, "\n")

# Poda el árbol con el cp óptimo
modelo_arbol_podado <- prune(modelo_arbol, cp = cp_optimo)

# Visualiza el árbol podado
rpart.plot(modelo_arbol_podado, type = 4, extra = 101,
           main = "Árbol Podado — actual_productivity_score")

# Predicciones del árbol podado sobre el conjunto de prueba
pred_arbol <- predict(modelo_arbol_podado, newdata = test)

cat("── Métricas Árbol en test ──\n")
rmse_arbol <- rmse(test$actual_productivity_score, pred_arbol)
mae_arbol  <- mae(test$actual_productivity_score,  pred_arbol)
cat("RMSE Árbol:    ", rmse_arbol, "\n")
cat("MAE  Árbol:    ", mae_arbol,  "\n")

#MODELO NULO (BASELINE)

baseline      <- rep(mean(train$actual_productivity_score), nrow(test))
rmse_baseline <- rmse(test$actual_productivity_score, baseline)
mae_baseline  <- mae(test$actual_productivity_score,  baseline)

#TABLA COMPARATIVA FINAL

cat("\n══════════════════════════════════════════════\n")
cat("   COMPARACIÓN DE MODELOS — test (n = 6.000)\n")
cat("══════════════════════════════════════════════\n")
cat(sprintf("%-20s %10s %10s\n", "Modelo", "RMSE", "MAE"))
cat(sprintf("%-20s %10.4f %10.4f\n", "Baseline (nulo)",  rmse_baseline, mae_baseline))
cat(sprintf("%-20s %10.4f %10.4f\n", "Árbol de decisión", rmse_arbol,   mae_arbol))
cat(sprintf("%-20s %10.4f %10.4f\n", "MLR",              rmse_mlr,      mae_mlr))
cat("══════════════════════════════════════════════\n")

# Gráfico de barras comparativo (RMSE de los tres modelos)
modelos <- c("Baseline", "Árbol", "MLR")
rmses   <- c(rmse_baseline, rmse_arbol, rmse_mlr)

barplot(rmses,
        names.arg = modelos,
        col       = c("#CBD5E1", "#1C7293", "#02C39A"),
        main      = "Comparación RMSE — test set",
        ylab      = "RMSE",
        ylim      = c(0, max(rmses) * 1.2))

text(x      = barplot(rmses, plot = FALSE),
     y      = rmses + 0.03,
     labels = round(rmses, 3),
     cex    = 0.9,
     font   = 2)

