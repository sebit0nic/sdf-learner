class SDFVisualizer:
    @staticmethod
    def init_progress_bar():
        print('   Progress: ' + 100 * '.', end='')

    @staticmethod
    def update_progress_bar(progress):
        print('\r   Progress: ' + (int(progress * 100) * '#') + (100 - int(progress * 100)) * '.', end='', flush=True)

    @staticmethod
    def end_progress_bar():
        print('\r   Progress: ' + 100 * '#', end='', flush=True)
        print('')
