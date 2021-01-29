import tensorflow as tf
import matplotlib 
matplotlib.use('Agg')
from contrastive_learning.data.get_dataset import get_dataset
from contrastive_learning.losses import get_contrastive_loss
from contrastive_learning.train_contrastive import train_contrastive
from contrastive_learning.models.resnet import ResNetSimCLR
from contrastive_learning.train_finetune import finetune
import neptune

neptune.init(project_qualified_name='Serre-Lab/paleo-ai',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiOTQzMzk1YTQtZmQxNS00MGI0LTg1YWUtMTU3ZWM2ZDE3NTVjIn0=',
             )

# Create experiment
logger= neptune.create_experiment('contrastive-pnas')

PATH= '/cifs/data/tserre_lrs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100/'
DATASET_SUP = 'pnas_sup'
DATASET_UNSUP = 'pnas_unsup'

if __name__ == "__main__":
    print('Trainig contrastive ')

    loss = get_contrastive_loss(temperature=1.0)
    batch_size = 32
    
    train, val, test = get_dataset(DATASET_UNSUP, batch_size=batch_size,path=PATH)
    
    input_shape = (128, 128, 3)
    out_dim = batch_size*2
    contrastive_model = ResNetSimCLR(input_shape,out_dim )        

    #contrastive_model = train_contrastive(train, val, test,projector, loss, tf.keras.optimizers.Adam(), epochs=1, verbose=True, froze_backbone=True,neptune=neptune)
    
    print('Fine Tune step ')

    train, val, test = get_dataset(DATASET_SUP, batch_size=batch_size,path=PATH)
    test_score = finetune(train, val, test, 19,
                      contrastive_model, epochs=2, verbose=True, froze_backbone=True,neptune=logger)
    