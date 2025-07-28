# src/connex/__init__.py

# Optionally, add logging or other initialization code here
import logging

# Configure logging at the package level
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

package_version = "0.1.0"

# Placeholder imports (until modules are populated)
from .examples.generate_example_data import *  # Use this to import everything from generate_data module
from .analysis.analysis import *
from .analysis.graph_network import *  # Use this to import everything from graph_builder module
from .analysis.metrics import *  # Use this to import everything from metrics module
from .plot.plot import *  # Use this to import everything from plot module
#from .preproc.preprocessing import *  # Use this to import everything from preprocessing module
#from .preproc.shapefile_tools import *  # Use this to import everything from shapefile_tools module
#from .utils.utils import *  # Use this to import everything from utils module


# If you want to limit what is exposed when using wildcard import (*),
# you can define the __all__ list (you can modify this once modules are populated):

# __all__ = ['function1', 'function2', 'class1', 'class2']
