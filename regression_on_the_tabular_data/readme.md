### Create conda environmental

```bash
conda create --name regression python=3.10.6
conda activate regression
# install all neccessary dependencies from requirements.txt
pip install --upgrade pip && pip install -r ./requirements.txt
```