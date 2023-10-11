import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from lazycodr.utils import generate_pr, get_pr_diff

console = Console()


app = typer.Typer()


@app.command()
def generate(repo_name: str, pr_number: int):
    pr_template = ''
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Getting diff...", total=None)
        pr_diff, pr = get_pr_diff(repo_name, pr_number)
        console.print(pr_diff)

        progress.add_task(description="Generating PR description...", total=None)
        res = generate_pr(pr_diff, pr_template)

        md = Markdown(res)
        console.print(md, width=90)

        pr.create_issue_comment(res)


if __name__ == "__main__":
    app()
