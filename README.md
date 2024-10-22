# MDHT OpenAI API with FastAPI


## Setup

1. If you don’t have Python installed, [install it from here](https://www.python.org/downloads/)

2. Clone this repository

3. Navigate into the project directory

   ```bash
   $ cd openai-quickstart-fastapi
   ```

4. Create a new virtual environment

   ```bash
   # Linux
   $ python -m venv venv
   $ . venv/bin/activate
   ```

   ```shell
   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

5. Install the requirements

   ```bash
   $ pip install -r requirements.txt
   ```

6. Make a copy of the example environment variables file

   ```bash
   # Linux
   $ cp .env.example .env
   ```

   ```shell
   # Windows
   xcopy .env.example .env
   ```

7. Add your [API key](https://beta.openai.com/account/api-keys) to the newly created `.env` file

8. Run the app

   ```bash
   $ python app.py
   or
   $ python main:app --reload
   ```
You should now be able to access the app at [http://localhost:5001](http://localhost:5001)! 

