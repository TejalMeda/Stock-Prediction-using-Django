# custom_filters.py

from django import template
from your_app.views import get_color_from_polarity

register = template.Library()

@register.filter
def get_polarity_color(polarity):
    return get_color_from_polarity(polarity)
