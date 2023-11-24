from typing import Dict, Type
from metrics.base_metric import BaseMetric
from metrics.distortion import DistortionMetric






# This maps the name of the metrics to the class of the metrics.
metrics_name_to_MetricsClass : Dict[str, Type[BaseMetric]] = {
    "distortion" : DistortionMetric,
}
