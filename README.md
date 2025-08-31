# RAG

# To run the project:
1. Clone this repository:
    ```sh
    git clone https://github.com/fano2458/RAG.git
    cd RAG/
    ```
2. Install the requirements:
   ```sh
   pip install -r requirements.txt
   ```
3. Make sure that ollama is installed on your PC (and gpt-oss:20b is downloaded) (installation for linux):
   ```sh
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull gpt-oss:20b
   ```
4. Run the app:
   ```sh
   streamlit run app.py
   ```
