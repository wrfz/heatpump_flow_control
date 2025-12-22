#!/bin/bash
# Code Quality Check Script für Heatpump Flow Control Integration
# Führt verschiedene Checks auf Python-Dateien aus

set -e

# Farben für Ausgabe
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Zähler für Ergebnisse
ERRORS=0
WARNINGS=0

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   Heatpump Flow Control - Code Quality Check${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Liste der zu prüfenden Dateien
FILES=(
    "custom_components/heatpump_flow_control/__init__.py"
    "custom_components/heatpump_flow_control/config_flow.py"
    "custom_components/heatpump_flow_control/const.py"
    "custom_components/heatpump_flow_control/flow_controller.py"
    "custom_components/heatpump_flow_control/number.py"
    "custom_components/heatpump_flow_control/switch.py"
)

# Check 1: Python Syntax
echo -e "${YELLOW}[1/4] Syntax-Check (py_compile)...${NC}"
SYNTAX_OK=true
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        if python3 -m py_compile "$file" 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} $file"
        else
            echo -e "  ${RED}✗${NC} $file - Syntax Error!"
            SYNTAX_OK=false
            ERRORS=$((ERRORS + 1))
        fi
    fi
done

if [ "$SYNTAX_OK" = true ]; then
    echo -e "${GREEN}✓ Syntax-Check passed${NC}\n"
else
    echo -e "${RED}✗ Syntax-Check failed${NC}\n"
fi

# Check 2: Ruff (Fast Linter)
echo -e "${YELLOW}[2/4] Ruff Linting...${NC}"
if command -v ruff &> /dev/null; then
    # Nur kritische Fehler: F (pyflakes - undefined vars, etc.)
    if ruff check custom_components/heatpump_flow_control/*.py --select=F --quiet 2>/dev/null; then
        echo -e "${GREEN}✓ Ruff check passed (no critical errors)${NC}\n"
    else
        echo -e "${RED}✗ Ruff found critical issues:${NC}"
        ruff check custom_components/heatpump_flow_control/*.py --select=F 2>&1
        echo ""
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${YELLOW}⚠ Ruff not installed (pip install ruff)${NC}\n"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 3: Pylint (Detailed Analysis)
echo -e "${YELLOW}[3/4] Pylint Analysis...${NC}"
if command -v pylint &> /dev/null; then
    PYLINT_OUTPUT=$(pylint --errors-only --disable=import-error custom_components/heatpump_flow_control/*.py 2>&1)
    if [ -z "$PYLINT_OUTPUT" ]; then
        echo -e "${GREEN}✓ Pylint check passed (no errors)${NC}\n"
    else
        echo -e "${RED}✗ Pylint found errors:${NC}"
        echo "$PYLINT_OUTPUT"
        echo ""
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${YELLOW}⚠ Pylint not installed (pip install pylint)${NC}\n"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 4: Import Test (optional - benötigt Home Assistant)
echo -e "${YELLOW}[4/4] Import Test (optional)...${NC}"
IMPORT_OK=true
for module in "__init__" "const" "flow_controller"; do
    if python3 -c "import sys; sys.path.insert(0, 'custom_components'); from heatpump_flow_control import $module" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $module"
    else
        echo -e "  ${YELLOW}⚠${NC} $module - Requires Home Assistant environment"
        IMPORT_OK=false
    fi
done

if [ "$IMPORT_OK" = true ]; then
    echo -e "${GREEN}✓ Import test passed${NC}\n"
else
    echo -e "${YELLOW}⚠ Import test skipped (Home Assistant not available)${NC}\n"
    WARNINGS=$((WARNINGS + 1))
fi

# Zusammenfassung
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Checks passed with $WARNINGS warning(s)${NC}"
    exit 0
else
    echo -e "${RED}✗ Found $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    exit 1
fi
