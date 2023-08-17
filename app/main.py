from flask import Flask, request
from pick_regno import pick_regno

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
async def predict():
    json_dict = request.get_json()

    result = pick_regno(
        camera_regno=json_dict['regno_recognize'],
        nn_regno=json_dict['afts_regno_ai'],
        camera_score=json_dict['recognition_accuracy'],
        nn_score=json_dict['afts_regno_ai_score'],
        nn_sym_scores=json_dict['afts_regno_ai_char_scores'],
        nn_len_scores=json_dict['afts_regno_ai_length_scores'],
        camera_type=json_dict['camera_type'],
        camera_class=json_dict['camera_class'],
        time_check=json_dict['time_check'],
        direction=json_dict['direction'],
        model_name='./micromodel.cbm'
    )

    return list(result)


if __name__ == '__main__':
    app.run()
