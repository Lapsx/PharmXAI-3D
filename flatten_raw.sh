#!/bin/bash
# ==============================================================================
# Script de Pré-processamento: Achatamento e Extração de Metadados
# ==============================================================================

echo "Iniciando extração e achatamento do PDBbind..."

# 1. Mapear exatamente quem são os complexos Refined ANTES de achatar e misturar
echo "Salvando a lista oficial do Refined Set para o Teste Cego..."
find raw/pbpp-2020 -mindepth 2 -maxdepth 2 -type d -exec basename {} \; > refined_ids.txt

# 2. Achatar tudo para a raiz do /raw
echo "Extraindo Refined Set (pbpp-2020)..."
find raw/pbpp-2020 -mindepth 2 -maxdepth 2 -type d -exec mv {} raw/ \; 2>/dev/null

echo "Extraindo General Set (P-L)..."
find raw/P-L -mindepth 2 -maxdepth 2 -type d -exec mv {} raw/ \; 2>/dev/null

# 3. Limpeza
echo "Limpando diretórios residuais..."
rm -rf raw/pbpp-2020 raw/P-L raw/P-L.tar.gz

echo "✅ Sucesso! O arquivo 'refined_ids.txt' foi gerado. O script train.py o utilizará para a separação hermética dos dados."
