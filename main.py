from models.neural import Neural
from models.neural_models.model_v1 import ModelsV1


def calibration_neural_model(args):

    neural_network_instance = create_classifier_model(args)
    neural_network_instance.calibration_neural_network()

def create_classifier_model(args):

    neural_model = Neural(args)

    if args.file_load_model != '':
        neural_model.load_model()

    if args.neural_model == "model_v1a":
        neural_model.create_neural_network(ModelsV1(args))

    return neural_model


def arguments_cmd_choice(args):

    if args.cmd == 'Calibration': calibration_neural_model(args)
    if args.cmd == 'CreateSamples': create_samples(args)
    if args.cmd == 'Training': training_neural_model(args)
    if args.cmd == 'Predict': predict_neural_model(args)
    if args.cmd == 'Analyse': evaluation(args)