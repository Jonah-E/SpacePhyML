from os import makedirs

from torch import load as torch_load

from .arcs.mms import PCReduced_arc, PCBaseline_arc
from ..utils.file_download import missing_files, download_file_with_status

class PCBaseline(PCBaseline_arc):
    """
    Load a train PCBaseline model for MMS dayside plasma region clessification.

    """
    _models = {'s42': {
                    'url': 'https://zenodo.org/records/15147451/files/model_PCBaseline_s42.pth?download=1',
                    'file': 'model_PCBaseline_s42.pth'},
               's84': {
                    'url': 'https://zenodo.org/records/15147451/files/model_PCBaseline_s84.pth?download=1',
                    'file': 'model_PCBaseline_s84.pth'},
               's168': {
                    'url': 'https://zenodo.org/records/15147451/files/model_PCBaseline_s168.pth?download=1',
                    'file': 'model_PCBaseline_s168.pth'},
               's336': {
                    'url': 'https://zenodo.org/records/15147451/files/model_PCBaseline_s336.pth?download=1',
                    'file': 'model_PCBaseline_s336.pth'},
               }

    def __init__(self, model='s42', path = './models'):
        """
        Examples:
            >>> from spacephyml.models.mms import PCBaseline
            >>> model = PCBaseline('s84')

        Args:
            model (string): The model to load, ['s42', 's84', 's168', 's336']
            path (string): Path to loacation of stored model.
        """
        super().__init__()

        if model not in self._models.keys():
            raise ValueError(f'Incorrect model, {model} not in' +
                             f'{self._models.keys()}')

        filepath = f'{path}/' + self._models[model]['file']

        missing = missing_files([filepath], './')
        if missing:
            print('Missing model file, downloading')
            makedirs(path, exist_ok=True)
            download_file_with_status(self._models[model]['url'], filepath)

        self.classifier.load_state_dict(torch_load(filepath, weights_only=True,
                                                    map_location=torch.device('cpu')))

class PCReduced(PCReduced_arc):
    """
    Load a train PCReduced model for MMS dayside plasma region clessification.

    """

    _models = {'s42': {
                    'url': 'https://zenodo.org/records/15147451/files/model_PCReduced_s42.pth?download=1',
                    'file': 'model_PCReduced_s42.pth'},
               's84': {
                    'url': 'https://zenodo.org/records/15147451/files/model_PCReduced_s84.pth?download=1',
                    'file': 'model_PCReduced_s84.pth'},
               's168': {
                    'url': 'https://zenodo.org/records/15147451/files/model_PCReduced_s168.pth?download=1',
                    'file': 'model_PCReduced_s168.pth'},
               's336': {
                    'url': 'https://zenodo.org/records/15147451/files/model_PCReduced_s336.pth?download=1',
                    'file': 'model_PCReduced_s336.pth'},
               }

    def __init__(self, model='s42', path = './models'):
        """
        Examples:
            >>> from spacephyml.models.mms import PCReduced
            >>> model = PCReduced('s84')

        Args:
            model (string): The model to load, ['s42', 's84', 's168', 's336']
            path (string): Path to loacation of stored model.
        """
        super().__init__()

        if model not in self._models.keys():
            raise ValueError(f'Incorrect model, {model} not in' +
                             f'{self._models.keys()}')

        filepath = f'{path}/' + self._models[model]['file']

        missing = missing_files([filepath], './')
        if missing:
            print('Missing model file, downloading')
            makedirs(path, exist_ok=True)
            download_file_with_status(self._models[model]['url'], filepath)

        self.classifier.load_state_dict(torch_load(filepath, weights_only=True,
                                                    map_location=torch.device('cpu')))
