import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
print("MODEL CREATED -*-")
visualizer = Visualizer(opt)
print("VISUALIZER CREATED -*-")

for i, data in enumerate(dataset):
    model.set_input(data)
    visualizer.display_current_results(model.get_current_visuals(), 1)
