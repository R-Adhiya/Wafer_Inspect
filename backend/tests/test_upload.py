"""Comprehensive tests for inspection image upload, validation, and retrieval."""

import io
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
from app.core.config import settings
from app.main import app

client = TestClient(app)


def create_synthetic_image(
    width: int = 128,
    height: int = 128,
    mode: str = "L",
    image_format: str = "PNG",
) -> bytes:
    """Generate deterministic synthetic image bytes."""
    img = Image.new(mode, (width, height), color=128 if mode in {"L", "I;16"} else (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format=image_format)
    return buf.getvalue()


def test_valid_png_upload():
    """Verify uploading a valid PNG returns 200 and accurate metadata."""
    img_bytes = create_synthetic_image(width=128, height=128, mode="L", image_format="PNG")
    response = client.post(
        "/inspection/upload",
        files={"test_image": ("wafer_sample.png", img_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "upload_id" in data
    assert data["test_image"]["original_filename"] == "wafer_sample.png"
    assert data["test_image"]["detected_format"] == "PNG"
    assert data["test_image"]["mode"] == "L"
    assert data["test_image"]["width"] == 128
    assert data["test_image"]["height"] == 128
    assert data["test_image"]["file_size_bytes"] == len(img_bytes)
    assert data["reference_image"] is None


def test_valid_jpg_upload():
    """Verify uploading a valid JPEG returns 200 and accurate metadata."""
    img_bytes = create_synthetic_image(width=100, height=100, mode="RGB", image_format="JPEG")
    response = client.post(
        "/inspection/upload",
        files={"test_image": ("wafer_rgb.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["test_image"]["detected_format"] == "JPEG"
    assert data["test_image"]["mode"] == "RGB"
    assert data["test_image"]["width"] == 100
    assert data["test_image"]["height"] == 100


def test_valid_tiff_upload():
    """Verify uploading a valid TIFF returns 200 and accurate metadata."""
    img_bytes = create_synthetic_image(width=120, height=80, mode="L", image_format="TIFF")
    response = client.post(
        "/inspection/upload",
        files={"test_image": ("wafer_gray.tif", img_bytes, "image/tiff")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["test_image"]["detected_format"] == "TIFF"
    assert data["test_image"]["mode"] == "L"
    assert data["test_image"]["width"] == 120
    assert data["test_image"]["height"] == 80


def test_valid_rgba_upload():
    """Verify uploading an RGBA image returns 200 and mode RGBA."""
    img = Image.new("RGBA", (96, 96), color=(100, 150, 200, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    response = client.post(
        "/inspection/upload",
        files={"test_image": ("wafer_rgba.png", img_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["test_image"]["detected_format"] == "PNG"
    assert data["test_image"]["mode"] == "RGBA"


def test_valid_16bit_upload():
    """Verify uploading a 16-bit grayscale image (I;16) is accepted."""
    img = Image.new("I;16", (128, 128), color=30000)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    response = client.post(
        "/inspection/upload",
        files={"test_image": ("wafer_16bit.png", img_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["test_image"]["mode"] in {"I;16", "I"}


def test_text_file_renamed_png():
    """Verify a text file with a .png extension is rejected with 415."""
    bad_bytes = b"This is not a real image header at all."
    response = client.post(
        "/inspection/upload",
        files={"test_image": ("fake_wafer.png", bad_bytes, "image/png")},
    )
    assert response.status_code == 415
    detail = response.json()["detail"]
    assert detail["code"] == "INVALID_FORMAT"


def test_zero_byte_file():
    """Verify a 0-byte file is rejected with 400."""
    response = client.post(
        "/inspection/upload",
        files={"test_image": ("empty_wafer.png", b"", "image/png")},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "FILE_EMPTY"


def test_oversized_file(monkeypatch):
    """Verify files exceeding MAX_UPLOAD_SIZE are rejected with 413."""
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE", 100)
    img_bytes = create_synthetic_image(width=128, height=128, mode="L", image_format="PNG")
    assert len(img_bytes) > 100

    response = client.post(
        "/inspection/upload",
        files={"test_image": ("large_wafer.png", img_bytes, "image/png")},
    )
    assert response.status_code == 413
    detail = response.json()["detail"]
    assert detail["code"] == "FILE_TOO_LARGE"


def test_truncated_png():
    """Verify a truncated image file is rejected with 422."""
    valid_bytes = create_synthetic_image(width=128, height=128, mode="L", image_format="PNG")
    # Cut off after PNG header but before image stream finishes
    truncated_bytes = valid_bytes[:45]

    response = client.post(
        "/inspection/upload",
        files={"test_image": ("corrupt.png", truncated_bytes, "image/png")},
    )
    # Either format check fails if too short or decode integrity fails
    assert response.status_code in {415, 422}
    detail = response.json()["detail"]
    assert detail["code"] in {"INVALID_FORMAT", "CORRUPT_IMAGE"}


def test_png_with_jpg_extension():
    """Verify a PNG image disguised with a .jpg extension is rejected with 415."""
    png_bytes = create_synthetic_image(width=128, height=128, mode="L", image_format="PNG")
    response = client.post(
        "/inspection/upload",
        files={"test_image": ("wafer_disguised.jpg", png_bytes, "image/jpeg")},
    )
    assert response.status_code == 415
    detail = response.json()["detail"]
    assert detail["code"] == "FORMAT_MISMATCH"


def test_unsupported_extension():
    """Verify unsupported file extensions are rejected with 415."""
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    bmp_bytes = buf.getvalue()

    response = client.post(
        "/inspection/upload",
        files={"test_image": ("wafer.bmp", bmp_bytes, "image/bmp")},
    )
    assert response.status_code == 415
    detail = response.json()["detail"]
    assert detail["code"] == "UNSUPPORTED_EXTENSION"


def test_path_traversal_filename():
    """Verify directory traversal filenames are sanitized and stored safely."""
    img_bytes = create_synthetic_image(width=128, height=128, mode="L", image_format="PNG")
    traversal_name = "../../etc/passwd.png"

    response = client.post(
        "/inspection/upload",
        files={"test_image": (traversal_name, img_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    upload_id = data["upload_id"]
    sanitized = data["test_image"]["original_filename"]
    assert "/" not in sanitized
    assert ".." not in sanitized

    # Confirm file on disk is under data/uploads/<upload_id>/test.png
    expected_path = settings.upload_path / upload_id / "test.png"
    assert expected_path.is_file()


def test_upload_with_and_without_reference():
    """Verify upload works both with and without a reference image."""
    test_bytes = create_synthetic_image(width=128, height=128, mode="L", image_format="PNG")
    ref_bytes = create_synthetic_image(width=256, height=256, mode="L", image_format="PNG")

    # 1. With reference (differing dimensions to check reference_warning)
    response_with_ref = client.post(
        "/inspection/upload",
        files={
            "test_image": ("test.png", test_bytes, "image/png"),
            "reference_image": ("ref.png", ref_bytes, "image/png"),
        },
    )
    assert response_with_ref.status_code == 200
    data_with_ref = response_with_ref.json()
    assert data_with_ref["reference_image"] is not None
    assert data_with_ref["reference_image"]["width"] == 256
    assert data_with_ref["reference_warning"] is not None

    # 2. Without reference
    response_no_ref = client.post(
        "/inspection/upload",
        files={"test_image": ("test.png", test_bytes, "image/png")},
    )
    assert response_no_ref.status_code == 200
    data_no_ref = response_no_ref.json()
    assert data_no_ref["reference_image"] is None
    assert data_no_ref["reference_warning"] is None


def test_get_image_serving_and_bad_upload_id():
    """Verify GET /inspection/upload/{upload_id}/{kind} and error handling for bad upload_id."""
    img_bytes = create_synthetic_image(width=128, height=128, mode="L", image_format="PNG")

    # Upload first
    upload_res = client.post(
        "/inspection/upload",
        files={"test_image": ("wafer.png", img_bytes, "image/png")},
    )
    upload_id = upload_res.json()["upload_id"]

    # Valid GET test image
    get_res = client.get(f"/inspection/upload/{upload_id}/test")
    assert get_res.status_code == 200
    assert get_res.headers["content-type"] == "image/png"
    assert len(get_res.content) == len(img_bytes)

    # Bad upload_id (malformed string)
    bad_id_res = client.get("/inspection/upload/not-a-valid-uuid/test")
    assert bad_id_res.status_code == 404
    assert bad_id_res.json()["detail"]["code"] == "INVALID_UPLOAD_ID"

    # Non-existent UUID
    non_existent_id_res = client.get("/inspection/upload/00000000-0000-0000-0000-000000000000/test")
    assert non_existent_id_res.status_code == 404
    assert non_existent_id_res.json()["detail"]["code"] == "UPLOAD_NOT_FOUND"

    # Invalid kind
    invalid_kind_res = client.get(f"/inspection/upload/{upload_id}/invalid_kind")
    assert invalid_kind_res.status_code == 400
    assert invalid_kind_res.json()["detail"]["code"] == "INVALID_KIND"
