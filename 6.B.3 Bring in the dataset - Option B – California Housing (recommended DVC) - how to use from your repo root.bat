:: activate your virtual environment first
call .venv\Scripts\activate

:: install the one extra dependency, if needed
pip install scikit-learn pandas

:: run the script
python mlops_assignment1\src\download_housing.py