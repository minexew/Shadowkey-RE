import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


logger = logging.getLogger(__name__)


def generate(output_dir: Path, **kwargs):
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent),
        autoescape=select_autoescape()
    )

    template = env.get_template("index.html")

    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "index.html"
    path.write_text(template.render(kwargs))
    logger.info("Wrote %s", path)
