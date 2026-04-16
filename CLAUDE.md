# Proyecto: Detección de Picos Emocionales en Llamadas de Call Center

## Formato
- LaTeX con IEEEtran.cls (IEEE journal, dos columnas)
- BibTeX (`bibliography/refs.bib`)
- Idioma: español

## Estructura del repositorio
- `latex/` — fuente LaTeX activa (contiene la versión más reciente compilable)
- `entregables/Entrega_N/` — copias acumulativas por entrega, cada una autocontenida
- `entregables/entrega{1,2,3}/` — versiones anteriores en ODT/PDF (no modificar)

## Comandos de compilación
```bash
cd entregables/Entrega_N
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Convenciones
- Un `\section{}` por apartado mayor; `\subsection{}` para A, B, C...
- Citas en formato `\cite{clave}`; todas las claves deben existir en `bibliography/refs.bib`.
- Tablas con `table*` para ocupar dos columnas cuando es grande.
- Figuras vectoriales con TikZ cuando sea posible (`tikz` + `positioning`).
- NO inventar citas bajo ninguna circunstancia. Si se necesita una cita nueva, buscarla
  y verificarla (DOI o URL institucional) antes de incluirla.

## Secciones del documento (orden en main.tex)
1. abstract
2. introduccion → `\section{Introducción}` (Entrega 6: expandir)
3. marco_teorico → `\section{Marco Teórico}` (Entrega 5: renombrar a "Referente Teórico" y expandir)
4. cronograma_objetivos → `\section{Cronograma de Objetivos}`
5. cronograma_trabajo → `\section{Cronograma de Trabajo}`
6. **metodologia** → `\section{Metodología}` (Entrega 4: NUEVA, antes de resultados_esperados)
7. resultados_esperados → `\section{Resultados Esperados}`
8. area_problematica → `\section{Área Problemática}`
9. justificacion → `\section{Justificación}`

## Versionado
- Cada `entregables/Entrega_N/` es autocontenida: debe compilar sola.
- Antes de crear `Entrega_N/`, copiar `Entrega_(N-1)/` completa (o `latex/` para Entrega 4).
- Git: un tag anotado por entrega (`entrega-4`, `entrega-5`, `entrega-6`).

## Qué NO tocar sin razón explícita
- Secciones ya aprobadas por el tutor (Entregas 0–3).
- Claves de citación ni labels de referencia cruzada existentes.
- La estructura de IEEEtran.cls.
- Carpeta `entregables/entrega{1,2,3}/` (versiones ODT).

## Qué SÍ se puede hacer
- Renombrar una sección existente si lo pide el plan de la entrega
  (ej: "Marco Teórico" → "Referente Teórico" en Entrega 5).
- Expandir subsecciones existentes con contenido adicional.
- Agregar figuras y tablas nuevas.
- Agregar paquetes LaTeX al preámbulo si son necesarios (tikz, etc.).
