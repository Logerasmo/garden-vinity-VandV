#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("normalize")


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # Глобальные настройки парсинга чисел
    "decimal_separators": [".", ","],
    "thousands_separators": [" ", "\u00a0", "_"],
    "strip_chars": [" ", "\u00a0", "\u202f", "\t"],

    # Правила нормализации по имени колонки (или по паре (колонка, единица))
    # Формат:
    #   "<column>": {
    #       "target_unit": "<unit>",
    #       "unit_column": "<optional column name with units>",
    #       "factors": { "<source_unit>": <float multiplier to target> , ... },
    #       "aliases": { "<alias>": "<canonical unit>" },
    #       "default_factor": <float or null>  # если единица не указана
    #   }
    "rules": {
        # --- Примеры для мощностей, энергий, времени и т.п. ---
        "power_w": {
            "target_unit": "W",
            "unit_column": "power_unit",
            "aliases": {
                "w": "W", "вт": "W",
                "kw": "kW", "квт": "kW",
                "mw": "MW", "мвт": "MW",
                "gw": "GW", "гвт": "GW",
            },
            "factors": {"W": 1.0, "kW": 1e3, "MW": 1e6, "GW": 1e9},
        },
        "time_s": {
            "target_unit": "s",
            "unit_column": "time_unit",
            "aliases": {
                "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s", "с": "s", "сек": "s",
                "min": "min", "mins": "min", "minute": "min", "minutes": "min", "мин": "min",
                "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h", "ч": "h", "час": "h",
            },
            "factors": {"s": 1.0, "min": 60.0, "h": 3600.0},
        },
        "length_m": {
            "target_unit": "m",
            "unit_column": "length_unit",
            "aliases": {
                "nm": "nm", "нм": "nm",
                "um": "um", "µm": "um", "мкм": "um",
                "mm": "mm", "мм": "mm",
                "cm": "cm", "см": "cm",
                "m": "m", "м": "m",
                "km": "km", "км": "km",
            },
            "factors": {"nm": 1e-9, "um": 1e-6, "mm": 1e-3, "cm": 1e-2, "m": 1.0, "km": 1e3},
        },
        "area_cm2": {
            "target_unit": "cm2",
            "unit_column": "area_unit",
            "aliases": {
                "cm2": "cm2", "cm^2": "cm2", "см2": "cm2",
                "mm2": "mm2", "mm^2": "mm2", "мм2": "mm2",
                "m2": "m2", "m^2": "m2", "м2": "m2",
            },
            "factors": {"mm2": 0.01, "cm2": 1.0, "m2": 1e4},
        },
        "voltage_v": {
            "target_unit": "V",
            "unit_column": "voltage_unit",
            "aliases": {
                "v": "V", "в": "V",
                "mv": "mV", "мв": "mV",
                "kv": "kV", "кв": "kV",
            },
            "factors": {"mV": 1e-3, "V": 1.0, "kV": 1e3},
        },
        "current_density_ma_cm2": {
            "target_unit": "mA/cm2",
            "unit_column": "current_density_unit",
            "aliases": {
                "ma/cm2": "mA/cm2", "ma cm-2": "mA/cm2", "ма/см2": "mA/cm2",
                "a/cm2": "A/cm2", "a cm-2": "A/cm2",
            },
            "factors": {"mA/cm2": 1.0, "A/cm2": 1e3},
        },
        "bandgap_ev": {
            "target_unit": "eV",
            "unit_column": "bandgap_unit",
            "aliases": {"ev": "eV", "эв": "eV", "mev": "meV", "мэВ": "meV"},
            "factors": {"meV": 1e-3, "eV": 1.0},
        },
    },
}


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    """
    Загружает конфигурацию правил нормализации.

    Поддерживаются файлы .json и .yaml/.yml (если установлен PyYAML).
    Если путь не указан — возвращается DEFAULT_CONFIG.

    Пример структуры config.json:

        {
          "decimal_separators": [".", ","],
          "thousands_separators": [" ", "_"],
          "rules": {
            "power_w": {
              "target_unit": "W",
              "unit_column": "power_unit",
              "aliases": { "kw": "kW", "квт": "kW" },
              "factors": { "W": 1.0, "kW": 1000.0 }
            }
          }
        }

    Args:
        path: путь к файлу конфигурации или None.

    Returns:
        Словарь с настройками (никогда не None).
    """
    if path is None:
        log.info("Конфигурация не указана — используется встроенная DEFAULT_CONFIG.")
        return DEFAULT_CONFIG

    path = Path(path)
    if not path.exists():
        log.warning("Конфиг %s не найден, используется DEFAULT_CONFIG.", path)
        return DEFAULT_CONFIG

    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "Для YAML-конфига установите PyYAML: pip install pyyaml"
                ) from exc
            with path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            raise ValueError(f"Неподдерживаемый формат конфига: {suffix}")
    except Exception as exc:
        log.error("Не удалось прочитать конфиг %s: %s. Используется DEFAULT_CONFIG.", path, exc)
        return DEFAULT_CONFIG

    # Мягкое слияние с дефолтными ключами, чтобы не терять глобальные настройки
    merged = {**DEFAULT_CONFIG, **cfg}
    merged["rules"] = {**DEFAULT_CONFIG.get("rules", {}), **cfg.get("rules", {})}
    log.info("Загружен конфиг: %s (правил: %d)", path, len(merged["rules"]))
    return merged


# ---------------------------------------------------------------------------
# Разбор чисел
# ---------------------------------------------------------------------------
_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?$")


def parse_number(raw: Any, cfg: Dict[str, Any]) -> Optional[float]:
    """
    Преобразует «грязное» значение в float.

    Обрабатывает:
        - пробелы, NBSP, табуляции, подчёркивания как разделители тысяч;
        - запятую и точку как десятичный разделитель;
        - знак +/-; экспоненциальную запись (1e-3);
        - пустые строки и None → None.

    Стратегия выбора десятичного разделителя:
        - если в строке есть и '.' и ',' — тот, что стоит ПРАВЕЕ, считается десятичным,
          левый — разделителем тысяч (типично для «1.234,56» и «1,234.56»);
        - если только один из них — он и есть десятичный.

    Args:
        raw: исходное значение (str, int, float, None).
        cfg: конфигурация (использует strip_chars, thousands_separators).

    Returns:
        float или None, если значение не распознано.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        # NaN проверяем отдельно
        try:
            if pd.isna(raw):
                return None
        except Exception:
            pass
        return float(raw)

    s = str(raw).strip()
    if not s:
        return None

    # Убираем валютные символы и распространённые единицы, оставляя только цифры/знаки
    # (единицы обычно выносятся в отдельную колонку, но бывает и внутри)
    s = s.replace("−", "-").replace("–", "-")

    # Удаляем пробелы-разделители тысяч
    for sep in cfg.get("thousands_separators", []):
        s = s.replace(sep, "")
    for ch in cfg.get("strip_chars", []):
        s = s.replace(ch, "")

    # Отрезаем всё, что после первого нечислового символа, кроме .,+-eE
    # (например, "100 kW" → "100")
    match = re.match(r"[+-]?[\d.,eE+\-]+", s)
    if not match:
        return None
    s = match.group(0)

    # Определяем десятичный разделитель
    has_dot = "." in s
    has_comma = "," in s
    if has_dot and has_comma:
        if s.rfind(",") > s.rfind("."):
            # «1.234,56» — точка тысяч, запятая десятичная
            s = s.replace(".", "").replace(",", ".")
        else:
            # «1,234.56» — запятая тысяч, точка десятичная
            s = s.replace(",", "")
    elif has_comma:
        # Запятая — десятичная, если после неё <= 2 цифр; иначе это тысячи
        # (эвристика; для большинства научных данных работает)
        tail = s.rsplit(",", 1)[-1]
        if len(tail) in (1, 2, 3) and not tail.startswith("0"):
            s = s.replace(",", ".")
        elif len(tail) <= 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")

    if not _NUMERIC_RE.match(s):
        return None

    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Нормализация значений
# ---------------------------------------------------------------------------
def _canonical_unit(unit: Optional[str], aliases: Dict[str, str]) -> Optional[str]:
    """Приводит обозначение единицы к каноническому виду через aliases (без учёта регистра)."""
    if unit is None:
        return None
    u = str(unit).strip().lower().replace(" ", "")
    if not u:
        return None
    return aliases.get(u, aliases.get(str(unit).strip(), None)) or aliases.get(u)


def normalize_value(
    value: Any,
    unit: Optional[Any],
    rule: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Нормализует одно значение согласно правилу.

    Args:
        value: исходное значение (число/строка).
        unit:  исходная единица (из отдельной колонки или из самой ячейки).
        rule:  правило из config["rules"][<name>].
        cfg:   глобальная конфигурация (для parse_number).

    Returns:
        (normalized_value, target_unit, error_message).
        Любой элемент может быть None.
    """
    target_unit = rule.get("target_unit")
    aliases: Dict[str, str] = rule.get("aliases", {})
    factors: Dict[str, float] = rule.get("factors", {})

    num = parse_number(value, cfg)
    if num is None:
        return None, target_unit, f"не удалось распознать число: {value!r}"

    # Единица: сначала из отдельного аргумента, иначе попробуем вытащить из строки value
    src_unit = _canonical_unit(unit, aliases)
    if src_unit is None and isinstance(value, str):
        # Ищем в строке известный алиас
        low = value.lower()
        for alias, canon in aliases.items():
            if re.search(rf"(?<![a-zа-я0-9]){re.escape(alias)}(?![a-zа-я0-9])", low):
                src_unit = canon
                break

    # Если единицы нет — используем default_factor (если задан), иначе считаем, что уже в целевой
    if src_unit is None:
        if "default_factor" in rule and rule["default_factor"] is not None:
            return num * float(rule["default_factor"]), target_unit, None
        # Не смогли определить — оставляем значение как есть, но помечаем
        return num, target_unit, f"единица не распознана (value={value!r}, unit={unit!r})"

    factor = factors.get(src_unit)
    if factor is None:
        return None, target_unit, f"нет коэффициента для единицы {src_unit!r}"

    return num * factor, target_unit, None


# ---------------------------------------------------------------------------
# Применение правил к DataFrame
# ---------------------------------------------------------------------------
def apply_rules(
    df: pd.DataFrame,
    rules: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Применяет правила нормализации к DataFrame.

    Режимы:
        inplace=False (по умолчанию):
            создаётся новая колонка <rule_name> с нормализованным числом
            и (при ошибках) колонка <rule_name>_errors. Исходные колонки не меняются.
        inplace=True:
            нормализованное число пишется ОБРАТНО в source_column (например, active_area),
            целевая единица — ОБРАТНО в unit_column (например, area_unit).
            Новые колонки не создаются. Ошибки только логируются.

    Args:
        df: исходный DataFrame.
        rules: словарь правил из конфига.
        cfg: глобальная конфигурация (для parse_number).
        inplace: режим «на месте» (см. выше).

    Returns:
        Новый DataFrame (копия).
    """
    out = df.copy()

    for rule_name, rule in rules.items():
        src_col = rule.get("source_column", rule_name)
        if src_col not in out.columns:
            continue

        unit_col = rule.get("unit_column")
        if unit_col and unit_col not in out.columns:
            unit_col = None

        target_unit = rule.get("target_unit")

        values: List[Optional[float]] = []
        errors: List[Optional[str]] = []

        for _, row in out.iterrows():
            raw_val = row[src_col]
            raw_unit = row[unit_col] if unit_col else None
            norm, _tgt, err = normalize_value(raw_val, raw_unit, rule, cfg)
            values.append(norm)
            errors.append(err)

        n_err = sum(1 for e in errors if e)

        if inplace:
            # 1) число — обратно в source_column (float64, NaN там, где ошибка)
            out[src_col] = pd.Series(values, index=out.index, dtype="float64")

            # 2) единица — обратно в unit_column, но только для успешных строк
            if unit_col is not None:
                new_units = [
                    (target_unit if v is not None else row[unit_col])
                    for v, row in zip(values, out.to_dict("records"))
                ]
                out[unit_col] = pd.Series(new_units, index=out.index, dtype="object")

            log.info(
                "Правило '%s' (inplace): %s ← нормализовано, %s ← '%s', ошибок: %d",
                rule_name, src_col, unit_col or "—", target_unit, n_err,
            )
            if n_err:
                log.warning(
                    "Правило '%s': %d значений не удалось нормализовать, оставлены как NaN в %s",
                    rule_name, n_err, src_col,
                )
        else:
            out[rule_name] = pd.Series(values, index=out.index, dtype="float64")
            if n_err:
                out[f"{rule_name}_errors"] = pd.Series(errors, index=out.index, dtype="object")
                log.warning(
                    "Правило '%s': %d ошибок, см. %s_errors",
                    rule_name, n_err, rule_name,
                )
            else:
                log.info("Правило '%s': все значения нормализованы успешно.", rule_name)

    return out
# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def read_csv_file(path: Path, chunksize: Optional[int] = None) -> Iterable[pd.DataFrame]:
    """
    Читает CSV-файл (целиком или чанками).

    Используется dtype=str, чтобы не терять ведущие нули и не портить «грязные» значения.
    Пустые ячейки превращаются в NaN только на этапе нормализации.

    Args:
        path: путь к CSV.
        chunksize: размер чанка (None — читаем целиком одним DataFrame).

    Yields:
        pd.DataFrame (один или несколько).
    """
    try:
        reader = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            encoding="utf-8-sig",  # учитывает BOM в начале файла
            chunksize=chunksize,
        )
    except UnicodeDecodeError:
        log.warning("UTF-8 не подошёл для %s, пробуем cp1251.", path)
        reader = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            encoding="cp1251",
            chunksize=chunksize,
        )

    if chunksize is None:
        yield reader  # type: ignore[arg-type]
    else:
        yield from reader  # type: ignore[arg-type]


def write_csv_file(df: pd.DataFrame, path: Path) -> None:
    """Записывает DataFrame в CSV (UTF-8 с BOM, чтобы Excel корректно открывал кириллицу)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log.info("Записан результат: %s (%d строк)", path, len(df))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _iter_input_files(target: Path) -> List[Path]:
    """Возвращает список CSV-файлов для обработки (файл или папка)."""
    if target.is_file():
        return [target]
    if target.is_dir():
        files = sorted(target.glob("*.csv"))
        if not files:
            log.warning("В папке %s не найдено CSV-файлов.", target)
        return files
    raise FileNotFoundError(f"Путь не найден: {target}")


def process_file(
    in_path: Path,
    out_path: Path,
    cfg: Dict[str, Any],
    chunksize: Optional[int],
    inplace: bool = False,
) -> None:
    """Обрабатывает один CSV-файл. При inplace=True правит те же колонки."""
    rules = cfg.get("rules", {})
    first_chunk = True

    for chunk in read_csv_file(in_path, chunksize=chunksize):
        normalized = apply_rules(chunk, rules, cfg, inplace=inplace)
        if first_chunk:
            normalized.to_csv(out_path, index=False, encoding="utf-8-sig", mode="w")
            first_chunk = False
        else:
            normalized.to_csv(out_path, index=False, encoding="utf-8-sig", mode="a", header=False)
    log.info("Готово: %s → %s (inplace=%s)", in_path.name, out_path, inplace)


def build_parser() -> argparse.ArgumentParser:
    """Собирает CLI-парсер."""
    p = argparse.ArgumentParser(
        description="Нормализация CSV: приведение единиц измерения к единому стандарту.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=Path, help="Входной CSV-файл или папка с CSV.")
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Выходная папка (по умолчанию — <входная>/normalized).",
    )
    p.add_argument(
        "-c", "--config", type=Path, default=None,
        help="Файл с правилами (config.json / config.yaml).",
    )
    p.add_argument(
        "--chunksize", type=int, default=None,
        help="Размер чанка для больших файлов (по умолчанию — читать целиком).",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Подробный лог (DEBUG).",
    )
    p.add_argument(
    "--inplace", action="store_true",
    help="Писать нормализованные значения обратно в исходные колонки "
         "(active_area + area_unit), не создавая новых колонок.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Точка входа CLI."""
    args = build_parser().parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = load_config(args.config)
    input_path: Path = args.input
    if not input_path.exists():
        log.error("Входной путь не существует: %s", input_path)
        return 2

    files = _iter_input_files(input_path)

    # Куда писать: если указан --output — туда, иначе рядом с исходником в normalized/
    if args.output is not None:
        out_dir = args.output
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        base = input_path if input_path.is_dir() else input_path.parent
        out_dir = base / "normalized"
        out_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    for f in files:
        try:
            process_file(f, out_dir / f.name, cfg, args.chunksize, inplace=args.inplace)
        except Exception as exc:  # noqa: BLE001
            log.exception("Ошибка при обработке %s: %s", f, exc)
            exit_code = 1

    log.info("Обработано файлов: %d, выходная папка: %s", len(files), out_dir)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

    #import pandas as pd
    '''df = pd.read_csv("data/normalized/devices.csv")
    print(df[["publication_id", "device_id", "active_area", "area_unit"]].head(10))
    print("Уникальные area_unit:", df["area_unit"].dropna().unique())
    print("Тип active_area:", df["active_area"].dtype)'''
