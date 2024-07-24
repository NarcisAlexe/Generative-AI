# Generative-AI
 
 Pasi pentru rulare


Pentru a rula acest proiect, trebuie sa aveti instalate urmatoarele dependente:

 - ollama: pentru interactiunea cu API-ul Ollama
 - openai: pentru integrarea cu OpenAI API
 - torch: pentru procesarea tensorilor
 - PyPDF2: pentru citirea fisierelor PDF
 - pyyaml: pentru manipularea fisierelor YAML
 - docx: pentru generarea documentelor Word

Acestea pot fi instalate folosind fisierul requirements.txt. Puteyi instala toate dependenyele cu comanda:

 - pip install -r requirements.txt

Odata descarcate dependentele, vor trebui descarcate modelele "mxbai-embed-large" si "llama3". 
Deschideti un terminal Windows PowerShell si rulati comenzile:

 - ollama pull mxbai-embed-large
 - ollama pull llama3

Dupa descarcarea acestora, navigati, prin intermediul terminalului, in directorul in care este salvat proiectul.
Rulati intai scriptul "upload.py" pentru a incarca exemplele de antrenare in fisierul "vault.txt". Puteti folosi comanda:

 - python upload.py
 
Dupa rulare, se va dechide un foarte simplu UI unde puteti selecta fisierele ce urmeaza a fi folosite pentru antrenare.
apasati butonul "Upload PDF's" si selectati fisierele PDF pe care doriti sa le folositi drept date de antrenare.

(* In directorul "train_data" veti gasi 16 documente PDF pe care le puteti folosi. *)

Dupa ce toate fisierele vor fi adaugate in "vault.txt", puteti inchide interfata, astfel incetand rularea programului.

Dupa ce fisierul "vault.txt" va fi populat, puteti rula script-ul "local_rag.py". Puteti folosi comanda:

 - python local_rag.py

Adresati cerinta in terminal si apasati "Enter". raspunsul generat va fi salvat sub numele "solicitarea_si_oferta.docx". 
Odata generat fisierul, tastati "exit" pentru a iesi din program.


 Antrenarea/manipularea si rularea AI-ului

 1. Colectare si Pregatire a Datelor:

 - Conversia Fisierelor:
   - PDF-uri: Conversia continutului fisierelor PDF in text folosind PyPDF2.
   - Fisiere Text: Incărcarea si normalizarea textului din fisierele text.
   - Fisiere JSON: Incărcarea si conversia datelor JSON intr-un format text.

 - Segmentare: Impartirea textului in "chunk-uri" de maximum 1000 caractere pentru a fi gestionabil si pentru a evita depasirea limitelor modelului AI.

 2. Generarea si Stocarea Embedding-urilor:

 - Crearea Embedding-urilor:
   - Generare: Utilizarea API-ului Ollama pentru a crea embedding-uri pentru fiecare segment de text din "vault.txt".
 - Stocare: Salvarea embedding-urilor intr-un format Tensor pentru utilizare ulterioara in cautari de context.

 3. Procesarea Intrebarilor:

 - Rescrierea Intrebarilor:
   - Context: Utilizarea modelului Ollama pentru a rescrie intrebarile pe baza contextului din istoricul conversatiei.
 - Obtinerea Contextului Relevant:
   - Calcul: Calcularea similaritatii cosinus intre embedding-urile intrebarii rescrise si embedding-urile din vault pentru a gasi cele mai relevante contexte.

 4. Generarea Raspunsurilor:

 - Raspuns pe Baza Contextului:
   - Integrare: Combinarea intrebarii cu contextul relevant obtinut si trimiterea catre modelul Ollama pentru generarea unui raspuns.
 - Interactie:
   - Conversare: Utilizarea raspunsurilor generate pentru a raspunde si adaugarea lor in istoricul conversatiei.

 5. Salvarea Documentelor:

 - Crearea Documentelor:
   - Document DOCX: Salvarea solicitarilor si raspunsurilor intr-un fisier DOCX folosind biblioteca docx, cu un format prestabilit.

 6. Initiere si Rulare:

 - Configurare API:
   - API Client: Configurarea clientului API pentru Ollama folosind URL-ul si cheia API.
 - Incarcare si Procesare:
   - Vault: Incărcarea si procesarea continutului din "vault.txt" si generarea embedding-urilor pentru toate segmentele de text.
 - Executie:
   - Loop: Rularea unui loop principal unde se primesc intrebari si se ofera raspunsuri bazate pe contextul relevat.