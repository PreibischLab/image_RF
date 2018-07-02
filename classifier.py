import os
import sys
import time
import glob
import logging
import warnings

import re

import numpy as np
import primfilters as pr

#import tifffile as tiff
# image read and write - takes tiff stacks, png, jpg
from skimage import io

# for resize the images:
from skimage._shared.utils import warn
from scipy import ndimage as ndi
from skimage.transform._geometric import AffineTransform
from skimage.transform import warp

from skimage.transform import resize

# for tests:
#from collections import Counter

# to save classifier (might not load in different versions):
import pickle

# r/w yaml
from ruamel.yaml import YAML

#import argparse
from argparse import Namespace

#import importlib
#importlib.reload(pr) # in case changes were made in the primfilters library - reload.

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score, f1_score

from skimage.morphology import remove_small_objects
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.measurements import label
"""
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('poster')
sns.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')
"""
timestr = time.strftime("%Y%m%d-%H%M%S")

# probably this should be an argument to the function
# def load_and_validate_params_from_yaml(yml_file_path)

yml_file_path = os.path.expanduser('~/Desktop/wings_for_RF/classifier/parameters.yaml')
yml_file_path = os.path.expanduser('~/Desktop/laura_test/classifier/parameters.yaml')
yml_file_path = '/scratch/AG_Preibisch/Ella/embryo_analysis/classifier/parameters.yaml'
yml_file_path = '/media/ella/DATA/embryo_analysis/classifier/parameters.yaml'

warnings.simplefilter("ignore")

##################################

import inspect
import hashlib
import gzip
import functools

class Store(object):
    def __init__(self, path):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def _get_path_for_key(self, key):
        return os.path.join(self.path, f'{key}.pkl.gz')

    def put(self, key, value):
        with gzip.open(self._get_path_for_key(key), 'wb') as f:
            pickle.dump(value, f, protocol=4)

    def get(self, key):
        try:
            with gzip.open(self._get_path_for_key(key), 'rb') as f:
                return pickle.load(f)
        except:
            os.remove(self._get_path_for_key(key))
            raise "Error occured"

    def __contains__(self, key):
        return os.path.isfile(self._get_path_for_key(key))

class memoize():
    def __init__(self, store_dir=os.path.join(os.getcwd(), '.memoize')):
        self.store = Store(store_dir)

    def __call__(self, func):
        def wrapped_f(*args, **kwargs):
            key = self._get_key(func, *args, **kwargs)
            if key not in self.store:
                val = func(*args, **kwargs)
                self.store.put(key, val)
            return self.store.get(key)
        functools.update_wrapper(wrapped_f, func)
        return wrapped_f


    def _arg_hash(self, *args, **kwargs):
        _str = pickle.dumps(args, 2) + pickle.dumps(kwargs, 2)
        return hashlib.md5(_str).hexdigest()

    def _src_hash(self, func):
        _src = inspect.getsource(func)
        return hashlib.md5(_src.encode()).hexdigest()

    def _get_key(self, func, *args, **kwargs):
        arg = self._arg_hash(*args, **kwargs)
        src = self._src_hash(func)
        return src + '_' + arg


##################################

def setup_logger(classifier_folder):

    # Handle loggings:
    
    #set different formats for logging output
    file_name = os.path.join(classifier_folder,f'classifier_{timestr}.log')
    console_logging_format = '%(levelname)s: %(pathname)s:%(lineno)s %(message)s'
    file_logging_format = '%(levelname)s: %(asctime)s: %(pathname)s:%(lineno)s %(message)s'
    
    # configure logger
    logging.basicConfig(level=logging.INFO, format=file_logging_format, filename=file_name)
    
    logger = logging.getLogger()
    # create a file handler for output file
    handler = logging.StreamHandler()
    # set the logging level for log file
    handler.setLevel(logging.WARNING)

    # create a logging format
    formatter = logging.Formatter(console_logging_format)
    handler.setFormatter(formatter)
    handler.setLevel(logging.ERROR)
    # add the handlers to the logger
    logger.addHandler(handler)

##################################

# Function to time each function:
def timeit(method):
    @functools.wraps(method)
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int(te - ts)
        else:
            logging.info('%r  %2.2f s' % (method.__name__, (te - ts)))
        return result
    return timed
###################################

def resize_downsample(image, output_shape, order=1, mode=None, cval=0, clip=True,
           preserve_range=False, anti_aliasing=True, anti_aliasing_sigma=None):
    """
    Resize image to match a certain size.
    Performs interpolation to up-size or down-size images. Note that anti-
    aliasing should be enabled when down-sizing images to avoid aliasing
    artifacts. For down-sampling N-dimensional images with an integer factor
    also see `skimage.transform.downscale_local_mean`.
    Parameters
    
    This code was copied from: 
    https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_warps.py#L34
    Commit Hush: 94b561e77aa551fa91c52d9140af220885e5181e
    Because anti-aliasing parameter in resize function was only introduced in skimage 0.15 which is still a dev version.
    Once 0.15 will become an oficial version and will be updated here, this function can be deleted from the code and just imported.
    I did not use downscale_local_mean because it only allows for downscaling parameter of int (no float) - output size might be very different than user requested.
    ----------
    image : ndarray
        Input image.
    output_shape : tuple or ndarray
        Size of the generated output image `(rows, cols[, ...][, dim])`. If
        `dim` is not provided, the number of channels is preserved. In case the
        number of input channels does not equal the number of output channels a
        n-dimensional interpolation is applied.
    Returns
    -------
    resized : ndarray
        Resized version of the input.
    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 1. The order has to
        be in the range 0-5. See `skimage.transform.warp` for detail.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.  The
        default mode is 'constant'.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
    anti_aliasing : bool, optional
        Whether to apply a Gaussian filter to smooth the image prior to
        down-scaling. It is crucial to filter when down-sampling the image to
        avoid aliasing artifacts.
    anti_aliasing_sigma : {float, tuple of floats}, optional
        Standard deviation for Gaussian filtering to avoid aliasing artifacts.
        By default, this value is chosen as (1 - s) / 2 where s is the
        down-scaling factor.
    Notes
    -----
    Modes 'reflect' and 'symmetric' are similar, but differ in whether the edge
    pixels are duplicated during the reflection.  As an example, if an array
    has values [0, 1, 2] and was padded to the right by four values using
    symmetric, the result would be [0, 1, 2, 2, 1, 0, 0], while for reflect it
    would be [0, 1, 2, 1, 0, 1, 2].
    Examples
    --------
    >>> from skimage import data
    >>> from skimage.transform import resize
    >>> image = data.camera()
    >>> resize(image, (100, 100), mode='reflect').shape
    (100, 100)
    """

    if mode is None:
        mode = 'constant'

    output_shape = tuple(output_shape)
    output_ndim = len(output_shape)
    input_shape = image.shape
    if output_ndim > image.ndim:
        # append dimensions to input_shape
        input_shape = input_shape + (1, ) * (output_ndim - image.ndim)
        image = np.reshape(image, input_shape)
    elif output_ndim == image.ndim - 1:
        # multichannel case: append shape of last axis
        output_shape = output_shape + (image.shape[-1], )
    elif output_ndim < image.ndim - 1:
        raise ValueError("len(output_shape) cannot be smaller than the image "
                         "dimensions")

    factors = (np.asarray(input_shape, dtype=float) /
               np.asarray(output_shape, dtype=float))

    if anti_aliasing_sigma is None:
        anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)
    else:
        anti_aliasing_sigma =             np.atleast_1d(anti_aliasing_sigma) * np.ones_like(factors)
        if np.any(anti_aliasing_sigma < 0):
            raise ValueError("Anti-aliasing standard deviation must be "
                             "greater than or equal to zero")

    image = ndi.gaussian_filter(image, anti_aliasing_sigma,
                                cval=cval, mode=mode)

    # 2-dimensional interpolation
    if len(output_shape) == 2 or (len(output_shape) == 3 and
                                  output_shape[2] == input_shape[2]):
        rows = output_shape[0]
        cols = output_shape[1]
        input_rows = input_shape[0]
        input_cols = input_shape[1]
        if rows == 1 and cols == 1:
            tform = AffineTransform(translation=(input_cols / 2.0 - 0.5,
                                                 input_rows / 2.0 - 0.5))
        else:
            # 3 control points necessary to estimate exact AffineTransform
            src_corners = np.array([[1, 1], [1, rows], [cols, rows]]) - 1
            dst_corners = np.zeros(src_corners.shape, dtype=np.double)
            # take into account that 0th pixel is at position (0.5, 0.5)
            dst_corners[:, 0] = factors[1] * (src_corners[:, 0] + 0.5) - 0.5
            dst_corners[:, 1] = factors[0] * (src_corners[:, 1] + 0.5) - 0.5

            tform = AffineTransform()
            tform.estimate(src_corners, dst_corners)

        out = warp(image, tform, output_shape=output_shape, order=order,
                   mode=mode, cval=cval, clip=clip,
                   preserve_range=preserve_range)

    else:  # n-dimensional interpolation
        coord_arrays = [factors[i] * (np.arange(d) + 0.5) - 0.5
                        for i, d in enumerate(output_shape)]

        coord_map = np.array(np.meshgrid(*coord_arrays,
                                         sparse=False,
                                         indexing='ij'))

        image = convert_to_float(image, preserve_range)

        ndi_mode = _to_ndimage_mode(mode)
        out = ndi.map_coordinates(image, coord_map, order=order,
                                  mode=ndi_mode, cval=cval)

        _clip_warp_output(image, out, order, mode, cval, clip)

    return out

###################################
"""
def show_image(im):
    plt.figure(figsize=(20,10))
    plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
    sns.despine(bottom=True, left=True)
"""
###################################

def load_and_validate_params_from_yaml():
    
    # Check if YAML file exists, if not throw error:
    if not os.path.isfile(yml_file_path):
        raise Exception(f'YAML parameters file does not exist in the classifier folder. Missing file: {yml_file_path}')
    
    # Load parameters from YAML:
    yaml=YAML(typ='safe')   # default, if not specfied, is 'rt' (round-trip)
    with open(yml_file_path) as yaml_file:
        user_params = yaml.load(yaml_file)

    # State expected parameters to compare with the file:
    keys_user_params, types_user_params = list(zip(*[
        ['project_folder', str],
        ['ims_file_prefix', str],
        ['channel_num', (int,bool)],
        ['is_relevance_mask', bool],
        ['c1_invert_values', bool],
        ['training_images', list],
        ['is_regression', bool],
        ['z_size', (float,int)],
        ['n_trees', int],
        ['n_cpu', int],
        ['bigger_dim_output_size', list],
        ['min_f1_score', float],
        ['what_to_run', str],
        ['classifier_file_name', (str, bool)],
        ['output_probability_map', bool],
        ['filters_params', dict]
        
    ]))
        
    # Verify that all the parameters exist in the file and that the type matches:
    for i,k in enumerate(keys_user_params):
        if k not in user_params:
            raise Exception(f'Expected parameter key name in YAML file:"{k}". Please fix the YAML file.') 
        if not isinstance(user_params[k],types_user_params[i]):
            raise Exception(f'YAML parameter "{k}" has to be of type {types_user_params[i]}. Please fix the variable in the YAML file.')

    # Important error check in case all params will be imported to the local variables.
    # e.g. If user entered in the YAML a parameter name that will conflict with a variable in the script.
    if len(user_params)>len(keys_user_params):
        raise Exception(f'expected exactly {len(keys_user_params)} parameters in YAML file. Please fix the YAMl file')
        
    # Verify : user_params['project_folder'] path exists:
    if not os.path.isdir(os.path.expanduser(user_params['project_folder'])):
        raise Exception('YAML parameter "project_folder" must be a path that exists. Please fix the path in the YAML file.')
    # Verify : user_params['training_images'] list elements are strings:
    if not all(isinstance(x,str) for x in user_params['training_images']):
        raise Exception('YAML parameter "training_images" list should only include strings. Please fix the variable in the YAML file.')
    # Verify : user_params['z_size'] is in range:
    if not 1>=user_params['z_size']>=0:
        raise Exception('YAML parameter "z_size" has to be 0<=z<=1. Please fix the value in the YAML file.')
    # Verify : user_params['n_cpu'] within possible range of number of CPUs:
    if not 32>=user_params['n_cpu']>0:
        raise Exception('YAML parameter "n_cpu" has to be 0<n<=32. Please fix the value in the YAML file.')
        
    # Verify : user_params['downsample_by'] and user_params['bigger_dim_output_size']:
    if user_params['bigger_dim_output_size']:
        if not all(isinstance(x,int) for x in user_params['bigger_dim_output_size']):
            raise Exception('All elements in YAML list parameter "output_size" must be int.')
            if any(x<8 for x in user_params['bigger_dim_output_size']):
                raise Exeption('All elements in YAML list parameter "output_size" must be bigger than 8')
        user_params['bigger_dim_output_size'].sort()
       
    # Verify user_params['min_f1_score']:
    if not 0.6<user_params['min_f1_score']<1:
        raise Exception('YAML min f1 score must be 0.6<f1<1. Please fix the YAML file.')

    # Verify user_params['filters_params']:
    filters_params = user_params['filters_params']
    # Define the type and range in the filter parameters - If list, define that it's list, the type in it, and the range.
    types_filters_params = {'gauss_sigma_range':[list,int,range(1,21)],'DoG_couples':[list,list,int],
                            'window_size_range':[list,int,range(2,11)],'aniso_diffus_n_iter':[int,range(1,51)],'aniso_diffus_conduc_coef':[int,range(20,101)],
                            'aniso_diffus_gamma':[float,np.arange(0,0.26,0.01)],'aniso_diffus_method':[int,range(1,3)],'gabor_freq':[list,float,(0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45)],'gabor_theta_range':[list,int,range(4)],
                            'frangi_scale_range':[list,list,int],'entropy_radius_range':[list,int,range(2,11)]}
    for f in filters_params:
        if f not in types_filters_params:
            raise Exception(f'YAML filters_params "{f}" is unexpected (maybe misspell?). Please fix the YAMl file.')
        if not isinstance(filters_params[f],types_filters_params[f][0]):
            raise Exception(f'YAML filters_params "{f}" should be of type {types_filters_params[f][0]}. Please fix the YAMl file.')
        if types_filters_params[f][0]==list:
            if not all(isinstance(x,types_filters_params[f][1]) for x in filters_params[f]):
                raise Exception(f'YAML filters_params "{f}" list should only hold elements of type {types_filters_params[f][1]}. Please fix the YAMl file.')
            if types_filters_params[f][1]==list:
                if not all(len(x)==2 for x in filters_params[f]):
                    raise Exception(f'YAML filters_params "{f}" internal lists should hold couples (two values exactly). Please fix the YAMl file.')
                if not all(isinstance(y,types_filters_params[f][2]) for x in filters_params[f] for y in x):
                    raise Exception(f'YAML filters_params "{f}" lists should only hold elements of type {types_filters_params[f][2]}. Please fix the YAMl file.')
            elif not all(x in types_filters_params[f][2] for x in filters_params[f]):
                logging.warning(filters_params[f])
                raise Exception(f'YAML filters_params "{f}" list elements should all by in {types_filters_params[f][2]}. Please fix the YAMl file.')
        elif filters_params[f] not in types_filters_params[f][1]:
            logging.warning(filters_params[f])
            raise Exception(f'YAML filters_params "{f}" should all by in {types_filters_params[f][1]}. Please fix the YAMl file.')

    # Put all user params in Namespase - just for readability and short variable names
    p = Namespace(**user_params)
    
    # Set "static" parameters:
    p.project_folder = os.path.expanduser(p.project_folder)
    p.data_folder = os.path.join(p.project_folder,'data')
    p.classifier_folder = os.path.join(p.project_folder,'classifier')
    p.temp_folder = os.path.join(p.classifier_folder,'temp_files')
    p.ims_folder = os.path.join(p.data_folder,'images')
    p.labels_folder = os.path.join(p.data_folder,'labels')
    if p.is_relevance_mask:
        p.relevance_masks_folder = os.path.join(p.data_folder,'relevance_masks')
    p.output_folder = os.path.join(p.data_folder,'output')
    os.makedirs(p.output_folder, exist_ok=True)
    
    # Set up cash folder:
    os.makedirs(p.temp_folder, exist_ok=True)
    
    # Reading images file names:
    p.all_imgs_files = [f for f in sorted(os.listdir(p.ims_folder)) if p.ims_file_prefix in f]
    p.all_files_timestamp = [os.path.getmtime(os.path.join(p.ims_folder,f)) for f in p.all_imgs_files]
    
    p.training_files = [aif for ti in p.training_images for aif in p.all_imgs_files if ti in aif] 
    if len(p.training_files)<len(p.training_images):
        print('need to uncomment it after testing')
        #print(list(zip(*[p.training_files, p.training_images])))
        #raise Exception('Not all training images requested are in images dir. Please fix YAML / add files to folder')
    p.training_files_timestamp = [[os.path.getmtime(os.path.join(p.ims_folder,t)),
                                   os.path.getmtime(os.path.join(p.labels_folder,t))] for t in p.training_files]
    if p.is_relevance_mask:
        p.training_files_timestamp = [p.training_files_timestamp] + [os.path.getmtime(os.path.join(p.relevance_masks_folder,t))
                                for t in p.training_files]
    
    
    return p
    
###################################
# Load image:
@timeit
def load_and_resize_image(path, big_dim_output_size, channel_num, is_mask=False):
    logging.info(path)
    im = (io.imread(path))
    
    if is_mask:
        max_val = np.max(im)
    
    # if not already - make the first dimension the z axis value
    if (im.ndim==3) and (im.shape[2]<im.shape[0]):
        im = np.swapaxes(im,0,2)
        im = np.swapaxes(im,1,2)
        
    if channel_num and is_mask==False:
        if im.ndim==3:
            # Channel_num values start at 1, python starts at 0:
            im = im[channel_num]
        if im.ndim==4:
            im = im[:,channel_num,:,:]
    
    org_im_shape = im.shape 
    
    # For mask images:
    #if is_mask:
    #    im = np.int8(im)
    if not is_mask:    
        im_max, im_min = im.max(), im.min()
        im = (im - im_min)/(im_max - im_min)

    # resize the image:
    # If image is 3D - only axis x,y will be downsampled:
    if big_dim_output_size:
        # Check that requested output size is smaller than input:
        if big_dim_output_size > max(im.shape):
            return im, False, True
        
        output_size = [big_dim_output_size, int(min(im.shape[-2:-1])/(max(im.shape)/big_dim_output_size))]
        if im.shape[-1]>im.shape[-2]:
            output_size = output_size[::-1]

        if im.ndim==2:
            im = resize_downsample(im,output_size, anti_aliasing=True)
        else:
            out_im = np.zeros([im.shape[0]] + output_size)
            for i,slic in enumerate(im):
                out_im[i] = resize_downsample(slic, output_size, anti_aliasing=True)
            im = out_im

    if is_mask:    
        im[im==np.max(im)] = max_val
        im[im!=np.max(im)] = 0
    
    return im, org_im_shape, False


# @memoize(store_dir=os.path.join(p.classifier_folder,f'os{p.output_size[l]}' if p.output_size[0] else f'ds{p.downsample_by[l]}'))
@timeit
def generate_all_filters(im_path, big_dim_output_size, z_size, file_name, file_timestamp, channel_num):

    """
    # All filters are - gaussian, sobel, prewitt, hessian, DoG, 
    # minimum, maximum, median, mean, variance,
    # anisotropic diffusion 2D, anisotropic diffusion 3D, 
    # bilateral, gabor, laplace, frangi (similar to structure), entrop (entropy)
    """
    
    im, org_im_shape, is_output_big = load_and_resize_image(os.path.join(im_path,file_name), big_dim_output_size, channel_num)
    
    if is_output_big:
        return False, 0, 0
    
    logging.info(f'image: {file_name}, shape: {im.shape}')
    
    if im.ndim==3:
        gauss_sigma_range = [[sig*z_size,sig,sig] for sig in p.filters_params['gauss_sigma_range']]
        DoG_couples = [np.swapaxes([i]*3,0,1)*[z_size,1,1] for i in p.filters_params['DoG_couples']]
    else:
        gauss_sigma_range = [[sig, sig] for sig in p.filters_params['gauss_sigma_range']]
        DoG_couples = [np.swapaxes([i]*2,0,1)*[1,1] for i in p.filters_params['DoG_couples']]
    
    # generate filters:
    
    filters = []
    
    logging.info('starting gauss')
    filters.extend([pr.gaussian(im, sig) for sig in gauss_sigma_range])
    
    gauss_filters = filters[:]
    
    logging.info('starting sobel')
    filters.extend([np.float32(pr.sobel(gauss_filters[i])) for i in range(len(gauss_filters))])
    
    logging.info('starting prewitt')
    filters.extend([np.float32(pr.prewitt(gauss_filters[i])) for i in range(len(gauss_filters))])    
    
    logging.info('starting hessian')
    filters.extend([np.float32(pr.hessian(gauss_filters[i])) for i in range(len(gauss_filters))])
    
    if im.ndim==3:
        logging.info('starting hessian zx')
        filters.extend([np.float32(np.swapaxes(pr.hessian(np.swapaxes(gauss_filters[i],0,1)),0,1)) 
                           for i in range(len(gauss_filters))])

        logging.info('starting hessianzy')
        filters.extend([np.float32(np.swapaxes(pr.hessian(np.swapaxes(gauss_filters[i],0,2)),0,2)) 
                           for i in range(len(gauss_filters))])
        
        logging.info('starting frangiyx')
        filters.extend([np.float32(np.swapaxes(pr.frangi(np.swapaxes(im,0,1), sig_range),0,1)) for sig_range in p.filters_params['frangi_scale_range']])

    logging.info('starting DoG')
    filters.extend([pr.difference_of_gaussians(im, dog) for dog in DoG_couples])

    # generate anisotropic diffusion filter
    # NEED TO IMPROVE THIS!
    logging.info('starting anisotropic_diffusion')
    filters.append(pr.anisotropic_diffusion(im))

    # generate bilateral filters
    logging.info('starting bilateral')
    filters.extend([np.float32(pr.bilateral(im, win_siz)) for win_siz in p.filters_params['window_size_range']])

    # generate minimum, maximum, median, mean, varience filters
    logging.info('starting min, max, med, mean, varience')
    for win_siz in p.filters_params['window_size_range']:
        filters.append(pr.minimum(im, win_siz))
        filters.append(pr.maximum(im, win_siz))
        filters.append(pr.median(im, win_siz))
        filters.append(pr.mean(im, win_siz))
        filters.append(pr.varience(im, win_siz))
    
    # generate gabor filter
    filter_name = 'gabor{g_list}thetas{t_list}'.format(g_list=''.join(str(i) for i in p.filters_params['gabor_freq']),t_list=''.join(str(i) for i in p.filters_params['gabor_theta_range']))
    logging.info(f'starting {filter_name}')
    for f in p.filters_params['gabor_freq']: 
        for t in p.filters_params['gabor_theta_range']:
            filters.append(pr.gabor(im, f, t/4.*(np.pi)))

    # generate frangi filter
    logging.info('starting frangi')
    filters.extend([np.float32(pr.frangi(im, sig_range)) for sig_range in p.filters_params['frangi_scale_range']])

    # generate entropy filter
    logging.info('starting entropy (with varying radiuses)')
    filters.extend([np.float32(pr.entrop(im, r)) for r in p.filters_params['entropy_radius_range']])

    return filters, org_im_shape, im.shape

# setting up the classifier matrices:
# X - all filters, Y - answers
@timeit
def train_classifier(p, loop_num):
    """
    parameters:
 
    """
    logging.info('Preparing  metrics to classifier')
    
    # Load label images:
    classes_ims = []
    relevance_ims = []
    for t in p.training_files:
        classes_ims.append(load_and_resize_image(os.path.join(p.labels_folder,t), p.bigger_dim_output_size[loop_num], False, True)[0])
        #classes_ims.append(load_and_resize_image(os.path.join(p.labels_folder,t), p.bigger_dim_output_size[loop_num], False, True)[0]==True)
        if p.is_relevance_mask:  
            #relevance_ims.append(load_and_resize_image(os.path.join(p.relevance_masks_folder,t), p.bigger_dim_output_size[loop_num], False, True)[0]==True)
            relevance_ims.append(load_and_resize_image(os.path.join(p.relevance_masks_folder,t), p.bigger_dim_output_size[loop_num], False, True)[0])
            
    
    if not p.is_relevance_mask:
        relevance_ims = [np.ones(c.shape)*255 for c in classes_ims]

    # Original images filters array:
    X = np.array([])
    # Classes array:
    c = np.array([])
    # Relevance array:
    r = np.array([])

    # Stack filters file and label images:
    for i,t in enumerate(p.training_files):
        temp_i = (generate_all_filters(p.ims_folder, p.bigger_dim_output_size[l], p.z_size, t, p.training_files_timestamp[i], p.channel_num))[0]
        # If requested output size bigger than image:
        if not temp_i:
            if loop_num!=0:
                logging.warning('Requested output size is bigger than original image. '                 'saving/using classifier from previous iteration although min_f1_score is not met.')
                return 0, 0, 0
            else:
                raise Exception('Requested output size is bigger than original image. Please decrese bigger_dim_output_size in user_params.')
        
        temp_i = np.vstack([f.flatten() for f in temp_i]).T
        X = np.vstack((X,temp_i)) if X.size else temp_i
        c = np.hstack((c, classes_ims[i].flatten())) if c.size else classes_ims[i].flatten()
        r = np.hstack((r, relevance_ims[i].flatten())) if r.size else relevance_ims[i].flatten()
        
    if c.shape[0]!=X.shape[0] or r.shape[0]!=X.shape[0]:
        raise Exception('label images size must be equal to original images size')
    
    X = np.vstack([x.flatten() for x in X])
    y = np.array([c for c in c.flatten()])
    r = np.array([r for r in r.flatten()])
    
    X = X[r==255]
    y = y[r==255]

    skf = StratifiedKFold(n_splits=3)
    
    classification_reports = list()
    f1_scores = list()
    
    # In case the user asked for regression and not classification:
    if not p.is_regression:
        for train_ix, test_ix in skf.split(X, y): # for each of K folds
            # define training and test sets
            X_train, X_test = X[train_ix,:], X[test_ix,:]
            y_train, y_test = y[train_ix], y[test_ix]

            # Train classifier
            clf = RandomForestClassifier() #(n_jobs=2) not sure if works on cluster..
            clf.fit(X_train, y_train)

            # Predict test set labels
            y_hat = clf.predict(X_test)
            classification_reports.append(classification_report(y_test, y_hat))
            f1_scores.append(f1_score(y_test, y_hat, average=None))

        print(*classification_reports, sep='/n', flush=True)

        # Train classifier
        clf = RandomForestClassifier() #(n_jobs=2) not sure if works on cluster..
        
    else:
        clf = RandomForestRegressor()
        
    clf.fit(X, y)
    
    stop_loop_bool = True if (np.mean(f1_scores)>p.min_f1_score) else False

    return clf, stop_loop_bool

@timeit
def save_classifier(clf, folder ,classifier_file_name):
    with open(os.path.join(folder, f'{classifier_file_name}'), 'wb') as f:
        pickle.dump(clf, f)

@timeit
def load_classifier(folder, classifier_file_name):
    with open(os.path.join(folder, classifier_file_name), 'rb') as f:
        return pickle.load(f)

# Apply classifier:
@timeit
def apply_classifier(p, clf, i_f, i, l):
    filters, org_im_shape, im_shape = generate_all_filters(p.ims_folder, p.bigger_dim_output_size[l], p.z_size, i_f, p.all_files_timestamp[i], p.channel_num)
    X_predict = np.vstack([f.flatten() for f in filters]).T
    if p.output_probability_map:
        yhat = clf.predict_proba(X_predict)
        
        predicted_im_downsampled = np.reshape(yhat[:,0], im_shape)
        out_im = resize(predicted_im_downsampled, org_im_shape).astype("float32")
    else:
        yhat = clf.predict(X_predict)
        predicted_im_downsampled = np.reshape(yhat, im_shape)
        out_im = resize(predicted_im_downsampled, org_im_shape).astype("uint8")

    if p.c1_invert_values==True:
        out_im = (np.max(out_im)-out_im)
        
    if p.is_relevance_mask:
        relevance_mask = load_and_resize_image(os.path.join(p.relevance_masks_folder,i_f), p.bigger_dim_output_size[l], False, True)[0]
        relevance_mask = resize(relevance_mask, org_im_shape).astype("uint8")
        out_im[relevance_mask==0] = 0
    
    return out_im

""" deleted function - because of multiclass - might add in future
def binarize_output(out_im):
    labeled_out, num_features = label(out_im)
    out_im = out_im>0.5
    out_im = remove_small_objects(out_im, 50, connectivity=1)
    return binary_fill_holes(out_im) #structure=[[[1]*3]*3]*3)
"""

#########################################################################
# MAIN

# Load + set parameters from yaml:
p = load_and_validate_params_from_yaml()

# Redefining the function, as the memoize decorator needs parameters - p:
generate_all_filters = memoize(store_dir=p.temp_folder)(generate_all_filters)
train_classifier = memoize(store_dir=p.temp_folder)(train_classifier)

setup_logger(p.classifier_folder)

if p.what_to_run in ['classify','both','Classify','Both','c', 'b','C','B']:
    # Loop through sizes of downsampling - for fast classification:
    for l in range(len(p.bigger_dim_output_size)):
        # Load images + labels then train classifier:
        clf, stop_loop_bool = train_classifier(p, l)

        if stop_loop_bool==True:
            break
        if not clf:
            l=l-1
            break;

    if stop_loop_bool==False:
        logging.warning('min_f1_score was not met using the current settings.')

    clf = train_classifier(p, l)[0]
    classifier_name = f'bdos{p.bigger_dim_output_size[l]}_ntrees{p.n_trees}_{timestr}.pkl'

    save_classifier(clf, p.classifier_folder, classifier_name)

if p.what_to_run in ['predict','Predict','p','P']:
    clf = load_classifier(p.classifier_folder, p.classifier_file_name)
    p.bigger_dim_output_size = [int((re.findall("bdos(\d+)", p.classifier_file_name))[0])]
    l=0

############# PREDICT ##############

if p.what_to_run in ['predict','both','Predict','Both','p', 'b','P','B']:
    for i, im_f in enumerate(p.all_imgs_files):
        out_im = apply_classifier(p, clf, im_f, i, l)
        io.imsave(os.path.join(p.output_folder,f'{im_f}'), out_im)