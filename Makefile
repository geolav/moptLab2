.PHONY: pdf clean help clean-all

# Основной целевой файл
TEX_FILE ?= MetOptLab2.tex
PDF_FILE = $(TEX_FILE:.tex=.pdf)
AUX_FILES = $(TEX_FILE:.tex=.aux) $(TEX_FILE:.tex=.log) $(TEX_FILE:.tex=.out) $(TEX_FILE:.tex=.toc) $(TEX_FILE:.tex=.fls) $(TEX_FILE:.tex=.fdb_latexmk)

# Компилятор (pdflatex или xelatex)
COMPILER ?= pdflatex
COMPILER_OPTS = -interaction=nonstopmode -shell-escape

help:
	@echo "📚 LaTeX Makefile для $(TEX_FILE)"
	@echo ""
	@echo "Команды:"
	@echo "  make pdf              - Компилировать в PDF (по умолчанию $(TEX_FILE))"
	@echo "  make pdf TEX_FILE=file.tex - Компилировать другой файл"
	@echo "  make clean            - Удалить вспомогательные файлы"
	@echo "  make clean-all        - Удалить все, включая PDF"
	@echo "  make help             - Эта справка"
	@echo ""
	@echo "Примеры:"
	@echo "  make pdf TEX_FILE=report.tex"
	@echo "  make pdf COMPILER=xelatex  (для документов с математикой)"
	@echo ""

pdf:
	@echo "🔨 Компилирование $(TEX_FILE) в PDF..."
	$(COMPILER) $(COMPILER_OPTS) $(TEX_FILE)
	@if [ -f $(PDF_FILE) ]; then \
		echo "✅ PDF успешно создан: $(PDF_FILE)"; \
	else \
		echo "❌ Ошибка при компилировании"; \
		exit 1; \
	fi

clean:
	@echo "🧹 Удаление вспомогательных файлов..."
	rm -f $(AUX_FILES)
	@echo "✅ Готово"

clean-all: clean
	@echo "🗑️  Удаление всех файлов, включая PDF..."
	rm -f $(PDF_FILE)
	@echo "✅ Готово"
