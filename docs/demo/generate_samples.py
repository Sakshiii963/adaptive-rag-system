"""Generate the synthetic PDF corpus used by the local demo."""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas


OUTPUT = Path(__file__).parent


def write_pdf(filename: str, title: str, paragraphs: list[str]) -> None:
    canvas = Canvas(str(OUTPUT / filename), pagesize=letter)
    width, height = letter
    for index, paragraph in enumerate(paragraphs):
        if index:
            canvas.showPage()
        canvas.setTitle(title)
        canvas.setFont("Helvetica-Bold", 18)
        canvas.drawString(72, height - 80, title)
        canvas.setFont("Helvetica", 11)
        y = height - 125
        for line in paragraph.split("\n"):
            canvas.drawString(72, y, line)
            y -= 20
    canvas.save()


write_pdf(
    "adaptive-retrieval-brief.pdf",
    "Adaptive Retrieval Brief",
    [
        "Adaptive retrieval evaluates the quality of retrieved evidence before an answer is generated.",
        "When confidence is below the configured threshold, the planner adds deterministic evidence terms, retries retrieval, and stops after a bounded retry budget.",
    ],
)
write_pdf(
    "verification-policy.pdf",
    "Verification Policy",
    [
        "Audit records are retained for seven years from the date of record creation.",
        "Incident escalation begins with the on-call engineer and moves to the service owner for unresolved severity-one incidents.",
    ],
)
