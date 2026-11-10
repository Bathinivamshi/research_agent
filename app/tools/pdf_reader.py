
import fitz


class PDFReaderService:

    def __init__(self):

        self.max_file_size = 20 * 1024 * 1024  # 20 MB
        self.max_pages = 500

    def extract_text(self, pdf_bytes: bytes) -> dict:

        # Step 1: Validate file size
        if not pdf_bytes:
            raise ValueError("Uploaded PDF is empty")

        if len(pdf_bytes) > self.max_file_size:
            raise ValueError("PDF exceeds the 20 MB size limit")

        # Step 2: Validate PDF signature
        if not pdf_bytes.startswith(b"%PDF-"):
            raise ValueError("Invalid PDF file")

        # Step 3: Open PDF
        try:

            document = fitz.open(
                stream=pdf_bytes,
                filetype="pdf"
            )

        except Exception as error:

            raise ValueError(
                f"Unable to open PDF: {error}"
            ) from error

        try:

            # Step 4: Check encryption
            if document.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported")

            # Step 5: Check page count
            total_pages = document.page_count

            if total_pages == 0:
                raise ValueError("PDF contains no pages")

            if total_pages > self.max_pages:
                raise ValueError("PDF exceeds the 500-page limit")

            # Step 6: Extract text page by page
            pages = []

            for page_number, page in enumerate(document, start=1):

                page_text = page.get_text("text").strip()

                pages.append({

                    "page_number": page_number,

                    "content": page_text,

                    "character_count": len(page_text)

                })

            # Step 7: Extract metadata
            metadata = document.metadata or {}

            # Step 8: Combine all page content
            full_text = "\n\n".join(
                page["content"]
                for page in pages
                if page["content"]
            )

            # Step 9: Return structured document
            return {

                "filename": metadata.get("title") or "Uploaded PDF",

                "author": metadata.get("author") or "Unknown",

                "total_pages": total_pages,

                "total_characters": len(full_text),

                "full_text": full_text,

                "pages": pages

            }

        finally:

            document.close()


pdf_reader_service = PDFReaderService()
# TODO: Extract text and metadata from PDF documents.
