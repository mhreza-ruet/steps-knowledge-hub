from typing import Any


def build_schema() -> dict[str, Any]:
    return {
        "name": "page_table_extraction",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "page_summary": {"type": "string"},
                "relevant_tables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "table_id": {"type": "string"},
                            "table_title_or_caption": {"type": "string"},
                            "table_type": {
                                "type": "string",
                                "enum": ["KINETICS", "ISOTHERM", "PARAM_SUMMARY", "MIXED", "UNKNOWN"]
                            },
                            "relevance_score": {"type": "number"},
                            "selection_reason": {"type": "string"},
                            "raw_headers": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "canonical_columns": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "raw_header": {"type": "string"},
                                        "canonical_name": {"type": "string"},
                                        "unit": {"type": "string"},
                                        "mapping_confidence": {"type": "number"}
                                    },
                                    "required": [
                                        "raw_header",
                                        "canonical_name",
                                        "unit",
                                        "mapping_confidence"
                                    ]
                                }
                            },
                            "rows": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "row_index": {"type": "integer"},
                                        "cells": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "additionalProperties": False,
                                                "properties": {
                                                    "raw_header": {"type": "string"},
                                                    "value": {"type": "string"}
                                                },
                                                "required": ["raw_header", "value"]
                                            }
                                        }
                                    },
                                    "required": ["row_index", "cells"]
                                }
                            },
                            "notes": {"type": "string"}
                        },
                        "required": [
                            "table_id",
                            "table_title_or_caption",
                            "table_type",
                            "relevance_score",
                            "selection_reason",
                            "raw_headers",
                            "canonical_columns",
                            "rows",
                            "notes"
                        ]
                    }
                },
                "ignored_tables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "table_id": {"type": "string"},
                            "reason_ignored": {"type": "string"}
                        },
                        "required": ["table_id", "reason_ignored"]
                    }
                }
            },
            "required": ["page_summary", "relevant_tables", "ignored_tables"]
        },
        "strict": True
    }