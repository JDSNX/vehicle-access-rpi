import argparse

from modules import RegisterDriver, TrainDriver, VehicleAccess

parser = argparse.ArgumentParser(description="The study is based on the concepts and existing studies about image processing that covers facial recognition and object detection. The researchers use Raspberry Pi 4B for the programming work of the study and a good quality camera for facial recognition and object detection.")
parser.add_argument('-v','--video_channel', help='[0] for internal camera | [1] for external camera', default=0)
parser.add_argument('-f','--classes-file', help='Path of classes file', required=True)
parser.add_argument('-w','--modeL-w', help='Path for weight models', required=True)
parser.add_argument('-c','--model-cfg', help='Path for configuration model', required=True)
parser.add_argument('-k','--face-cascade', help='Path for haarcascade `.xml`', required=True)
parser.add_argument('-r','--recognizer', help='Path for trainner.yml file', required=True)
parser.add_argument('-t','--helmet', help='Path for output helmet photo', required=True)
parser.add_argument('-p','--pickles', help='Path for .pickle file', required=True)
parser.add_argument('-x','--music_path', help='Path for music folder', required=True)
parser.add_argument('-s','--func', default='main', const='main', nargs='?', choices=['main', 'train', 'register'], help='List of functions to run: main, train (default: %(default)s)')
args = vars(parser.parse_args())

if __name__ == '__main__':

    va = VehicleAccess(
        args['video_channel'],
        args['classes_file'],
        args['model_w'],
        args['model_cfg'],
        args['face_cascade'],
        args['recognizer'],
        args['helmet'],
        args['music_path']
    )

    if args['func'] == 'main':
        va.main()
        print("[INFO] Done...")
    elif args['func'] == 'train':
        td = TrainDriver(args['face_cascade'])
        print("[INFO] Training done...")
    elif args['func'] == 'register':
        rd = RegisterDriver(
            args['video_channel'],
            args['face_cascade']
        )
        rd.register()
        print("[INFO] Register done...")