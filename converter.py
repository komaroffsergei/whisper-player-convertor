#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Консольная утилита для конвертации моделей Whisper (HF формата) в CTranslate2 (ct2) для использования в whisperx/faster-whisper.

Пример:
  python converter.py /path/to/hf_model /path/to/output_ct2 --quantization int8_float16

Требования:
  - Установлен пакет ctranslate2, предоставляющий утилиту `ct2-transformers-converter`
    pip install ctranslate2

Подсказки:
  - Если модель содержит LoRA-адаптеры, рекомендуется предварительно слить их в веса и сохранить цельный чекпоинт
    (как делается в нашем скрипте finetune_whisper.py при флаге --merge_lora).
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def build_copy_files_list(model_dir: Path):
    """Собирает список файлов, которые безопасно копировать, если они присутствуют в директории модели.
    Возвращает список ИМЁН файлов (а не путей), как того требует ct2-transformers-converter.
    """
    candidates = [
        "tokenizer.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocabulary.json",
        "special_tokens_map.json",
        "processor_config.json",
    ]
    present = []
    for name in candidates:
        if (model_dir / name).exists():
            present.append(name)
    return present


def find_converter_executable():
    """Находит исполняемый файл конвертера ct2-transformers-converter.
    Возвращает команду (список строк) для запуска конвертера.
    """
    exe = shutil.which("ct2-transformers-converter")
    if exe:
        return [exe]
    # Фолбэк через Python-модуль, если системный бинарь не найден
    return [sys.executable, "-m", "ctranslate2.converters.transformers"]


def _has_any_weight_file(model_dir: Path) -> bool:
    patterns = [
        "pytorch_model.bin",
        "pytorch_model-*.bin",
        "model.safetensors",
        "*.safetensors",
        "tf_model.h5",
        "model.ckpt.index",
        "flax_model.msgpack",
    ]
    for patt in patterns:
        if any(model_dir.glob(patt)):
            return True
    return False


def convert(model_path: Path, output_dir: Path, quantization: str = "int8_float16"):
    if not model_path.exists():
        raise FileNotFoundError(f"Не найден путь к модели: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Проверка прав записи (важно для окружений типа Kaggle)
    real_out = output_dir.resolve()
    if str(real_out).startswith("/kaggle/input"):
        raise OSError(f"Путь только для чтения: {real_out}. Укажите директорию внутри /kaggle/working")

    # Проверим наличие весов модели
    if not _has_any_weight_file(model_path):
        print(
            "Ошибка: в директории модели не найдены весовые файлы (pytorch_model*.bin, *.safetensors, tf_model.h5, model.ckpt.index, flax_model.msgpack).\n"
            "Возможные причины и решения:\n"
            "- Вы передали папку с LoRA-адаптером, но не выполнили merge в базовую модель.\n"
            "  Решение: выполните merge (например, скриптом fine_tuning/finetune_whisper.py с --merge_lora или fine_tuning/merge_lora.py).\n"
            "- Вы передали путь к CTranslate2-модели (ct2), а нужен HF-чекпоинт.\n"
            "  Решение: укажите путь к HF-модели (директория с config.json и весами).\n",
            file=sys.stderr,
        )
        sys.exit(2)

    copy_files = build_copy_files_list(model_path)
    cmd = find_converter_executable()
    cmd += [
        "--model", str(model_path),
        "--output_dir", str(output_dir),
        "--quantization", quantization,
        "--force",
    ]
    if copy_files:
        cmd += ["--copy_files", *copy_files]

    print("Выполняем конвертацию в CTranslate2:")
    print(" ", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка конвертера: {e}", file=sys.stderr)
        sys.exit(e.returncode)

    print(f"Готово. CTranslate2 модель сохранена в: {output_dir}")


def main():
    ap = argparse.ArgumentParser(description="Конвертация Whisper (HF) в CTranslate2")
    ap.add_argument("model", type=str, help="Путь к модели HF (директория с config.json, pytorch_model*.bin и т.д.)")
    ap.add_argument("output", type=str, help="Папка для сохранения CTranslate2 модели")
    ap.add_argument("--quantization", type=str, default="int8_float16",
                    choices=["float32", "float16", "int8", "int8_float16", "int8_float32"],
                    help="Схема квантования (см. ct2-transformers-converter --help)")
    args = ap.parse_args()

    model_path = Path(args.model)
    output_dir = Path(args.output)

    convert(model_path, output_dir, args.quantization)


if __name__ == "__main__":
    main()
