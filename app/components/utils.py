

def output_fields_from_question_sets(question_sets):
    fields = []
    for q in question_sets:
        field_type_map = {
            "single_label": "Single-label",
            "multi_label": "Multi-label",
            "boolean": "Boolean",
            "score": "Score",
            "text": "Text",
            "ranking": "Ranking",
        }

        fields.append(
            {
                "name": q.get("field_name", ""),
                "type": field_type_map.get(q.get("field_type", "single_label"), "Single-label")
            }
        )
    return fields