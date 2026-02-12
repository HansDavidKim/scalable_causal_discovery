# Argument Parsing is much easier with typer
from typer import Typer

app = Typer()

@app.command()
def hello_world():
    print('Hello, World!')

if __name__ == '__main__':
    app()