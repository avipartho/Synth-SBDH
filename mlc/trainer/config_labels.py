label2index = {
    'sbdh_gpt4_v2_multilabel':
        {
            'label_barriers_to_care': 0,
            'label_substance_abuse': 1,
            'label_housing_insecurity': 2,
            'label_financial_insecurity': 3,
            'label_psychiatric_symptoms_or_disorders': 4,
            'label_isolation_or_loss_of_relationship': 5,
            'label_patient_disability': 6,
            'label_violence': 7,
            'label_legal_problems': 8,
            'label_transitions_of_care': 9,
            'label_pain': 10,
            'label_food_insecurity': 11,
        },
    'sbdh_gpt4_msf_multilabel':
        {
            'label_barriers_to_care': 0,
            'label_substance_abuse': 1,
            'label_housing_insecurity': 2,
            'label_financial_insecurity': 3,
            'label_psychiatric_symptoms_or_disorders': 4,
            'label_isolation_or_loss_of_relationship': 5,
            'label_patient_disability': 6,
            'label_violence': 7,
            'label_legal_problems': 8,
            'label_transitions_of_care': 9,
            'label_pain': 10,
            'label_food_insecurity': 11,
        },
    'sbdh_gpt4_msf_v3_multilabel':
        {
            'label_barriers_to_care': 0,
            'label_substance_abuse': 1,
            'label_housing_insecurity': 2,
            'label_financial_insecurity': 3,
            'label_psychiatric_symptoms_or_disorders': 4,
            'label_isolation_or_loss_of_relationship': 5,
            'label_patient_disability': 6,
            'label_violence': 7,
            'label_legal_problems': 8,
            'label_transitions_of_care': 9,
            'label_pain': 10,
            'label_food_insecurity': 11,
        },
    'mimic_sbdh':{
        'label_substance_abuse': 0,
        'label_housing_insecurity': 1,
        'label_financial_insecurity': 2,
        'label_isolation_or_loss_of_relationship': 3,
    }
}


# index2prompt = {
#     i: f"Any indication of '{' '.join(sbdh.split('_')[1:])}'?" for sbdh,i in label2index.items()
# }
