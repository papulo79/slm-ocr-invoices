# Diseño del Sistema de Optimización de Prompts con GEPA

> **Documento de especificación de alto nivel**  
> **Propósito:** Compartible entre múltiples IAs, independiente de implementación técnica  
> **Estado:** Implementado y validado  
> **Última actualización:** 2026-04-09

---

## 1. Contexto y Motivación

### Proyecto base
Sistema de OCR para facturas en PDF que utiliza:
- **Modelo:** Ministral 3B (SLM local via LM Studio, API OpenAI-compatible)
- **Input:** Imágenes de facturas (PDF convertido a PNG, 300 DPI)
- **Output:** JSON estructurado según esquema Factur-X (`schema.json`)
- **Prompt activo:** `prompt.txt` (versión optimizada)

### Problemas identificados (estado inicial)
El sistema funcionaba correctamente en general, pero presentaba errores sistemáticos:

| Problema | Ejemplo | Impacto |
|----------|---------|---------|
| **Confusión NIF** | NIF "B83834747" extraído como "883834747" (la B se confunde con 8) | Datos fiscales incorrectos |
| **Descripciones truncadas** | Descripciones de líneas de factura incompletas, UUIDs cortados | Pérdida de información |
| **Campos vacíos ocasionales** | `buyer.vat_id` vacío en facturas OVH | Datos incompletos |
| **Seller/buyer confundidos** | En facturas de autónomos, emisor y receptor invertidos | Error estructural |

### Por qué GEPA
El prompt engineering manual es:
- **Tedioso:** Requiere iteraciones manuales de prueba y error
- **Inconsistente:** Cambios basados en intuición, no en datos
- **No escalable:** Difícil mantener múltiples prompts optimizados

GEPA (Genetic-Pareto Prompt Evolution) automatiza este proceso mediante algoritmos evolutivos que:
1. Evalúan el prompt contra un dataset de casos de prueba
2. Identifican patrones de error específicos
3. Generan variantes de prompt basadas en esos errores
4. Seleccionan las mejores variantes mediante criterios de Pareto

---

## 2. Objetivos del Sistema

### Objetivo principal
Implementar un pipeline de optimización automática de prompts que mejore la precisión del OCR de facturas sin intervención manual iterativa.

### Objetivos específicos

1. **Dataset Golden**
   - Dataset de 24 facturas representativas (implementado)
   - Cada factura tiene su correspondiente JSON "ground truth" (`gold.json`, corregido manualmente)
   - Cobertura: NIFs de persona física y empresa, descripciones con UUIDs, facturas multi-página, autónomos, proveedores cloud

2. **Evaluador Automático**
   - Comparar JSON extraído vs ground truth
   - Métricas por campo (no solo global)
   - Detección específica de problemas conocidos (B vs 8, truncamiento)

3. **Optimizador GEPA**
   - Explorar automáticamente variantes del prompt
   - Presupuesto configurable (`--iterations`), con parada por estancamiento
   - Criterio de aceptación: mejora estricta sobre el mejor score conocido
   - Protección contra degradación: rechaza prompts >150% tamaño semilla

4. **Gestión por modelo**
   - Prompts optimizados organizados en `results/{modelo}/best_prompt.txt`
   - `prompt.txt` como versión activa (elegida manualmente)
   - Compatible con cualquier modelo OpenAI-compatible en LM Studio

---

## 3. Arquitectura Conceptual

### Flujo de datos

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FASE 1: PREPARACIÓN                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Facturas PDF  ──►  Ground Truth Manual  ──►  Dataset Golden            │
│  (24 archivos)      (JSON corregidos)        (fuente de verdad)         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FASE 2: EVALUACIÓN                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Prompt Variante  ──►  OCR (Ministral 3)  ──►  JSON Predicho            │
│       │                                               │                 │
│       │                                               ▼                 │
│       │                                      ┌─────────────────┐       │
│       └──────────────────────────────────────│   EVALUADOR     │       │
│                                              │                 │       │
│  Ground Truth ◄──────────────────────────────│  • Comparación  │       │
│       │                                      │    campo a campo│       │
│       │                                      │  • Score NIF    │       │
│       └──────────────────────────────────────│  • Score Desc   │       │
│                                              │  • Score Total  │       │
│                                              └─────────────────┘       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    FASE 3: OPTIMIZACIÓN (GEPA)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Iteración N:                                                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │ 1. EVALUAR  │───►│ 2. REFLEJAR │───►│ 3. PROPO-   │                  │
│  │   (score)   │    │  (errores)  │    │    NER      │                  │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                  │
│                                               │                         │
│                                               ▼                         │
│                                        ┌─────────────┐                  │
│                                        │ 4. ACEPTAR  │                  │
│                                        │  (si mejora)│                  │
│                                        └──────┬──────┘                  │
│                                               │                         │
│                                               ▼                         │
│  Iteración N+1:                        ┌─────────────┐                  │
│  (repetir hasta                        │ Prompt N+1  │                  │
│   convergencia)                        └─────────────┘                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FASE 4: RESULTADO                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Prompt Óptimo  ──►  Validación contra dataset  ──►  Deploy            │
│  (mejor versión)      holdout (opcional)                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Componentes principales

| Componente | Responsabilidad | Entradas | Salidas |
|------------|-----------------|----------|---------|
| **Dataset Manager** | Gestionar las facturas y sus ground truths | Directorio de PDFs + JSONs manuales | Dataset estructurado accesible por índice |
| **OCR Engine** | Extraer JSON de una factura usando un prompt | Imagen + Prompt variant | JSON predicho |
| **Evaluador** | Calcular score comparando predicción vs truth | JSON predicho + Ground truth | Scores por campo y global |
| **GEPA Core** | Orquestar el proceso evolutivo | Dataset + Prompt semilla + Configuración | Prompt optimizado + Logs de iteraciones |
| **Proposer** | Generar nuevas variantes de prompt | Errores detectados + Prompt actual | Propuesta de nuevo prompt |

---

## 4. Especificación del Evaluador

### Requisitos funcionales

El evaluador debe comparar un JSON predicho contra el ground truth y producir métricas estructuradas.

#### Métricas por campo

| Campo | Tipo de comparación | Peso relativo |
|-------|---------------------|---------------|
| `seller.vat_id` | Exactitud (case-insensitive, pero preservar letras/números) | Alto (crítico) |
| `seller.name` | Similitud textual (fuzzy matching) | Medio |
| `buyer.vat_id` | Exactitud | Alto |
| `buyer.name` | Similitud textual | Medio |
| `invoice_number` | Normalización + exactitud (ignorar prefijos como "F-") | Medio |
| `invoice_date` | Exactitud tras normalizar a YYYY-MM-DD | Medio |
| `amount` / `vat_amount` | Tolerancia de 0.01€ | Alto |
| `line_items[].description` | Similitud textual por línea | Alto (uno de los problemas principales) |
| `line_items[].unit_price` | Exactitud numérica | Medio |
| `line_items[].quantity` | Exactitud | Bajo |

#### Detección de errores específicos

El evaluador debe detectar y reportar:

1. **Error "B vs 8"**: Cuando un NIF que debería empezar con letra (A-H, J-N, P-S) comienza con número
2. **Error de truncamiento**: Cuando la descripción predicha es prefijo de la descripción real (indica corte prematuro)
3. **Error de campo vacío**: Cuando un campo obligatorio está ausente o vacío
4. **Error de tipo**: Valores numéricos donde debería haber strings o viceversa

### Output del evaluador

```json
{
  "file": "factura_001.pdf",
  "scores": {
    "seller_vat": 1.0,
    "seller_name": 0.95,
    "line_descriptions": 0.87,
    "amount": 1.0,
    "total": 0.91
  },
  "issues": [
    {
      "field": "seller.vat_id",
      "type": "ocr_confusion",
      "subtype": "letter_vs_number",
      "expected": "B83834747",
      "actual": "883834747",
      "severity": "high"
    },
    {
      "field": "line_items[2].description",
      "type": "truncation",
      "expected": "Servicio de consultoría avanzada...",
      "actual": "Servicio de consultoría",
      "severity": "medium"
    }
  ]
}
```

---

## 5. Especificación del Proceso GEPA

### Entradas

- **Prompt semilla:** El prompt actual (`prompt.txt`)
- **Dataset:** 10 facturas con ground truth
- **Configuración:**
  - Máximo de iteraciones: 50
  - Umbral de aceptación: Score nuevo > Score anterior (mejora estricta)
  - Tamaño de lote de evaluación: 10 (dataset completo)

### Ciclo de iteración

1. **Evaluar prompt actual**
   - Procesar las 10 facturas con el prompt actual
   - Calcular score global promedio
   - Identificar todos los errores

2. **Generar variante (Proposer)**
   - Analizar errores detectados
   - Proponer modificaciones específicas al prompt para corregir esos errores
   - Ejemplo de instrucción al proposer:
     > "El prompt actual confunde la letra 'B' con el número '8' en NIFs.  
     > Sugiere una modificación al prompt que enfatice que los NIF españoles  
     > empiezan con letra y que la 'B' debe distinguirse del '8'."

3. **Aceptar o rechazar**
   - Evaluar el nuevo prompt en el mismo dataset
   - Si score nuevo > score anterior: Aceptar y continuar
   - Si score nuevo ≤ score anterior: Rechazar, intentar otra variante

4. **Criterio de parada**
   - Número máximo de iteraciones alcanzado (50)
   - No se encuentra mejora en N iteraciones consecutivas (estancamiento)
   - Score perfecto alcanzado (improbable, pero posible)

### Salidas

- **Prompt optimizado:** Archivo de texto con el mejor prompt encontrado
- **Log de iteraciones:** Registro de cada iteración (prompt, score, errores)
- **Reporte final:** Comparativa antes/después, estadísticas de mejora

---

## 6. Estructura de Datos

### Organización de archivos

```
proyecto/
├── app.py                      # Interfaz Streamlit
├── batch_eval.py               # Procesamiento batch
├── config.yaml                 # Configuración (no versionado, ver config.yaml.example)
├── config.yaml.example         # Plantilla de configuración
├── prompt.txt                  # Prompt activo (elegido manualmente)
├── schema.json                 # Esquema JSON de salida (Factur-X)
│
├── gepa/                       # Módulo de optimización GEPA
│   ├── __init__.py
│   ├── adapter.py              # Interfaz OCR ↔ GEPA (carga imágenes, llama al modelo)
│   ├── evaluator.py            # Métricas campo a campo + detección de issues
│   ├── proposer.py             # Generador de variantes de prompt vía LLM externo
│   └── optimizer.py            # Orquestador del ciclo evolutivo
│
├── data/golden/                # Dataset de evaluación (24 muestras)
│   └── {nombre_factura}/
│       ├── page_01.png         # Imagen(es) de la factura
│       ├── predicted.json      # Predicción original del modelo (referencia)
│       └── gold.json           # Ground truth (corregido manualmente)
│
├── results/                    # Salidas de optimización, organizadas por modelo
│   └── {nombre-modelo}/
│       ├── best_prompt.txt     # Mejor prompt encontrado para ese modelo
│       ├── iterations/         # Log JSON de cada iteración
│       └── report.json         # Resumen de la optimización
│
└── scripts/
    ├── prepare_dataset.py      # Genera el dataset golden desde PDFs
    ├── evaluate_prompt.py      # Evalúa un prompt contra el dataset
    ├── run_gepa.py             # Ejecuta la optimización GEPA completa
    └── test_proposer.py        # Valida la conexión con el Proposer
```

### Formato del Dataset Golden

Cada factura en su propia carpeta con:
- **Imágenes:** `page_01.png`, `page_02.png`, … (una por página del PDF)
- **Predicción original:** `predicted.json` (generado por `prepare_dataset.py`, no editar)
- **Ground truth:** `gold.json` (copia de `predicted.json` corregida manualmente)

---

## 7. Criterios de Éxito

### Métricas objetivo

| Métrica | Valor inicial (v1) | Valor objetivo | Resultado obtenido |
|---------|-------------------|----------------|-------------------|
| Score global promedio | 0.8796 | ≥ 0.95 | **0.9441** (Ministral 3B) |
| Precisión NIF seller | ~0.80 | ≥ 0.95 | **1.000** |
| Precisión NIF buyer | ~0.70 | ≥ 0.95 | **1.000** |
| Precisión descripciones | ~0.75 | ≥ 0.90 | **0.90+** |
| Campos vacíos obligatorios | < 5% | 0% | ~0% en campos críticos |

### Validación cualitativa

Además de las métricas, se debe verificar que:
- El prompt optimizado no sea excesivamente largo (aumenta coste/tiempo)
- No introduzca overfitting (evaluar con 1-2 facturas de holdout si es posible)
- Mantenga o mejore la claridad de las instrucciones

---

## 8. Consideraciones y Restricciones

### Limitaciones conocidas

1. **Coste computacional:** Cada iteración procesa 24 facturas con el SLM
   - Tiempo real: ~8-9 minutos por iteración (GPU local, Ministral 3B, 300 DPI)
   - Total estimado: ~3h para 20 iteraciones

2. **Modelos reasoning:** Variantes con razonamiento interno (ej: `ministral-3-8b-reasoning`) funcionan peor para extracción estructurada que los modelos base instruct. No se recomienda su uso como OCR engine en este pipeline.

3. **Proposer:** La calidad del LLM externo afecta directamente la velocidad de convergencia. MiniMax M2.7 y GPT-4o/5.x funcionan bien. Los modelos reasoning del proposer incluyen bloques `<think>` que se limpian automáticamente.

4. **Context window:** El modelo OCR necesita ≥ 20 000 tokens de contexto configurados en LM Studio. Con el valor por defecto (~7 000) las facturas largas generan JSON incompleto.

5. **Degradación GEPA:** Si el proposer genera prompts que empeoran el score, el sistema lo rechaza y reintenta. Con ≥10 rechazos consecutivos (estancamiento), GEPA termina y guarda el mejor prompt conocido.

### Restricciones técnicas

- Mantener compatibilidad con LM Studio (API OpenAI-compatible)
- No añadir dependencias pesadas (proyecto debe seguir siendo ligero)
- Código en Python, consistente con el estilo existente

---

## 9. Futuras Extensiones (Opcional)

Estas funcionalidades están fuera del alcance inicial pero se consideran para versiones futuras:

1. **Optimización multi-prompt:** Optimizar prompts para diferentes tipos de factura simultáneamente
2. **Online learning:** Continuar optimizando basado en errores de producción
3. **Comparación A/B:** Framework para comparar prompts en producción
4. **Dataset expansible:** Mecanismo para añadir nuevas facturas al dataset golden

---

## 10. Glosario

| Término | Definición |
|---------|------------|
| **GEPA** | Genetic-Pareto Prompt Evolution - Algoritmo de optimización de prompts mediante evolución genética |
| **Ground Truth** | El resultado correcto/verdadero contra el que se compara la predicción |
| **NIF** | Número de Identificación Fiscal (identificador tributario español) |
| **OCR** | Optical Character Recognition - Reconocimiento óptico de caracteres |
| **Overfitting** | Sobreajuste - cuando el modelo memoriza casos específicos en lugar de generalizar |
| **Pareto** | Criterio de selección que mantiene soluciones no dominadas (ninguna es peor en todos los aspectos) |
| **Proposer** | Componente que genera nuevas variantes de prompt basándose en errores |
| **Score** | Puntuación numérica (0-1) que indica la calidad de la extracción |
| **SLM** | Small Language Model - Modelo de lenguaje pequeño (vs LLM) |

---

## Historial de Cambios

| Versión | Fecha | Autor | Cambios |
|---------|-------|-------|---------|
| 0.1 | 2026-04-08 | Usuario + IA | Documento inicial |
| 1.0 | 2026-04-09 | Usuario + IA | Actualización post-implementación: resultados reales, dataset 24 muestras, estructura de results por modelo, limitaciones conocidas |

---

**Nota para IAs:** Este documento describe el QUÉ y el POR QUÉ, no el CÓMO. La implementación técnica (clases, funciones, dependencias) debe diseñarse caso por caso manteniendo la compatibilidad con esta especificación.
