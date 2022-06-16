from flask import (
    Blueprint, render_template
)

from .__init__ import create_app

create_app()

bp = Blueprint('app', __name__)


@bp.route('/')
def index():
    return render_template('app/index.html')
