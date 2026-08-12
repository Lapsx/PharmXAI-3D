import os
import shutil
import tarfile

def print_instructions():
    print("="*60)
    print("🧬 COMO BAIXAR O PDBBIND 2020 REFINED SET (GRATUITAMENTE) 🧬")
    print("="*60)
    print("O site oficial do PDBbind exige login institucional.")
    print("Como você já usa GPUs do Kaggle para treinar o PINO, a melhor ")
    print("forma é usar a API do Kaggle que já tem o dataset hospedado!")
    print("\nPASSO 1: Certifique-se de que o kaggle está instalado:")
    print("   pip install kaggle")
    print("\nOPÇÃO A: Para teste rápido local (Refined Set - 5.000 complexos - ~2.5 GB)")
    print("   kaggle datasets download -d asaniczka/pdbbind-2020-refined-set")
    print("\nOPÇÃO B: Para o Treinamento Real (General Set - ~19.000 complexos - ~15 GB)")
    print("   * O General Set é imenso. Procure no Kaggle por 'pdbbind general set', ou")
    print("   * Registre-se oficialmente em http://www.pdbbind.org.cn/ para baixar os arquivos completos.")
    print("\nPASSO 3: Extraia o arquivo zip que foi baixado.")
    print("\nPASSO 4: Mova a pasta extraída (que contém as milhares de")
    print("subpastas como '1hsg', '3ln1', etc.) e renomeie-a para 'raw'")
    print("dentro da nossa pasta PharmXAI-3D.")
    print("\nFeito isso, é só rodar 'python3 dataset.py' e ir tomar um café!")
    print("A compilação vai demorar um pouquinho, mas criará o nosso .pt ultraleve.")
    print("="*60)

if __name__ == "__main__":
    print_instructions()
