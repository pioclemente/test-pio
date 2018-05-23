import invariant from 'invariant';
import React from 'react';
import PropTypes from 'prop-types';

import {
  construct,
  componentDidMount,
  componentDidUpdate,
  componentWillUnmount,
} from 'react-google-maps/lib/utils/MapChildHelper';

import { MAP, DRAWING_MANAGER } from 'react-google-maps/lib/constants';

/**
 * @url https://developers.google.com/maps/documentation/javascript/3.exp/reference#DrawingManager
 */
const ALL_SHAPE = '__SECRET_ALL_SHAPE_DO_NOT_USE_OR_YOU_WILL_BE_FIRED';
const BUTTON_CENTER = 'center';
const BUTTON_DRAW_POLYGON = 'draw_polygon';
const BUTTON_CIRCLE = 'draw_circle';
const BUTTON_SELECT_AREA = 'select_area';
const BUTTON_DELETE = 'delete';

const eventMap = {
  onCircleComplete: 'circlecomplete',
  onMarkerComplete: 'markercomplete',
  onOverlayComplete: 'overlaycomplete',
  onPolygonComplete: 'polygoncomplete',
  onPolylineComplete: 'polylinecomplete',
  onRectangleComplete: 'rectanglecomplete',
};

const updaterMap = {
  drawingMode(instance, drawingMode) {
    instance.setDrawingMode(drawingMode);
  },
  options(instance, options) {
    instance.setOptions(options);
  },
};

function deselectAllButtonAsync(enableDrawingMenuButtons) {
  setTimeout(
    () => {
      enableDrawingMenuButtons();
      document.querySelectorAll('.srpmapDraw').forEach((el) => el.classList.remove('drawmanager_button_selected'));
    },
    315,
  );
}

function deselectAllButton(enableDrawingMenuButtons) {
  enableDrawingMenuButtons();
  document.querySelectorAll('.srpmapDraw').forEach((el) => el.classList.remove('drawmanager_button_selected'));
}

export class DrawingManager extends React.PureComponent {
  
  static propTypes = {
    /**
     * @type OverlayType
     */
    defaultDrawingMode: PropTypes.any,

    /**
     * @type DrawingManagerOptions
     */
    defaultOptions: PropTypes.any,

    /**
     * @type OverlayType
     */
    drawingMode: PropTypes.any,

    /**
     * @type DrawingManagerOptions
     */
    options: PropTypes.any,

    /**
     * function
     */
    onCircleComplete: PropTypes.func,

    /**
     * function
     */
    onMarkerComplete: PropTypes.func,

    /**
     * function
     */
    onOverlayComplete: PropTypes.func,

    /**
     * function
     */
    onPolygonComplete: PropTypes.func,

    /**
     * function
     */
    onPolylineComplete: PropTypes.func,

    /**
     * function
     */
    onRectangleComplete: PropTypes.func,
  }

  static contextTypes = {
    [MAP]: PropTypes.object,
  }
  
  static buttons = {
    center: { id: 'drawmanager_center_map', s: 'srpmapDraw center drawmanager_button_selected', d: 'srpmapDraw center' },
    draw_polygon: { id: 'drawmanager_draw_polygon', s: 'srpmapDraw polygon drawmanager_button_selected', d: 'srpmapDraw polygon' },
    draw_circle: { id: 'drawmanager_draw_circle', s: 'srpmapDraw circle drawmanager_button_selected', d: 'srpmapDraw circle' },
    select_area: { id: 'drawmanager_draw_select_area', s: 'srpmapDraw select_area drawmanager_button_selected', d: 'srpmapDraw select_area' },
    delete: { id: 'drawmanager_delete', s: 'drawmanager_delete_selected btncancel', d: 'drawmanager_delete btncancel' },
  }
  
  /*
   * @url https://developers.google.com/maps/documentation/javascript/3.exp/reference#DrawingManager
   */
  constructor(props, context) {
    super(props, context);
    invariant(
      google.maps.drawing,
      'Did you include "libraries=drawing" in the URL?',
    );

    this.currentButton = undefined;
    const drawingManager = new google.maps.drawing.DrawingManager()
    construct(DrawingManager.propTypes, updaterMap, this.props, drawingManager)
    drawingManager.setMap(this.context[MAP]);
    this.state = {
      [DRAWING_MANAGER]: drawingManager,
      [ALL_SHAPE]: [],
    };
  }
  
  componentDidMount() {
    const drawingManager = this.state[DRAWING_MANAGER];

    google.maps.event.addListener(drawingManager, 'overlaycomplete', (event) => {
      deselectAllButton(this.props.enableDrawingMenuButtons);
      this.currentButton = undefined;
      this.state[DRAWING_MANAGER].setDrawingMode(null);
      this.state[ALL_SHAPE].push(event);
      const newShape = event.overlay;
      newShape.type = event.type;
      if (event.type !== 'marker') {
        this.props.setEditingShape(newShape);
      } else {
        this.activateAreaSelection();
      } 
    }); 

    google.maps.event.addDomListener(document.getElementById(DrawingManager.buttons[BUTTON_CENTER].id), 'click', () => {
      const button = event.target;
      button.className = DrawingManager.buttons[BUTTON_CENTER].s;
      this.context[MAP].setZoom(this.props.zoom);
      this.context[MAP].setCenter(new google.maps.LatLng(this.props.centerCoords.lat, this.props.centerCoords.lng));
      deselectAllButtonAsync(this.props.enableDrawingMenuButtons);
    });

    google.maps.event.addDomListener(document.getElementById(DrawingManager.buttons[BUTTON_DRAW_POLYGON].id), 'click', (event) => {
      const button = event.target;

      this.clickButtonActions(false);
      if (this.currentButton !== undefined && this.currentButton === BUTTON_DRAW_POLYGON) {
        this.currentButton = undefined;
        button.classList.remove('drawmanager_button_selected');
        this.props.enableDrawingMenuButtons();
        this.state[DRAWING_MANAGER].setDrawingMode(null);
        this.props.activeDrawing(null);
        this.props.onResumeMap();
      } else {
        this.currentButton = BUTTON_DRAW_POLYGON;
        button.classList.add('drawmanager_button_selected');
        this.props.disableDrawingMenuButtons();    
        this.deleteAllShape(this.currentButton);
        drawingManager.setDrawingMode(google.maps.drawing.OverlayType.POLYGON);
        this.props.activeDrawing('polygon');
        this.props.tracking('Interaction', 'ClickOnDraw');
      }
    });

    google.maps.event.addDomListener(document.getElementById(DrawingManager.buttons[BUTTON_CIRCLE].id), 'click', (event) => {
      const button = event.target;
      this.clickButtonActions(false);
      if (this.currentButton !== undefined && this.currentButton === BUTTON_CIRCLE) {
        this.currentButton = undefined;
        button.classList.remove('drawmanager_button_selected');
        this.props.enableDrawingMenuButtons();
        this.state[DRAWING_MANAGER].setDrawingMode(null);
        this.props.activeDrawing(null);
        this.props.onResumeMap();
      } else {
        this.currentButton = BUTTON_CIRCLE;
        button.classList.add('drawmanager_button_selected');
        this.props.disableDrawingMenuButtons();
        this.deleteAllShape(this.currentButton);
        drawingManager.setDrawingMode(google.maps.drawing.OverlayType.CIRCLE);
        this.props.activeDrawing('circle');
        this.props.tracking('Interaction', 'ClickOnRadius');
      }
    });

    google.maps.event.addDomListener(document.getElementById(DrawingManager.buttons[BUTTON_SELECT_AREA].id), 'click', (event) => {
      const button = event.target;
      if (this.currentButton !== undefined && this.currentButton === BUTTON_SELECT_AREA) {
        this.clickButtonActions(false);  
        this.currentButton = undefined;
        button.classList.remove('drawmanager_button_selected');
        this.props.enableDrawingMenuButtons();
        this.state[DRAWING_MANAGER].setDrawingMode(null);    
        this.props.activeDrawing(null);    
      } else {
        this.clickButtonActions(true);        
        this.activateAreaSelection(this.currentButton);
        this.props.activeDrawing('area');
        this.props.tracking('Interaction', 'ClickOnL7');
      }
    });

    google.maps.event.addDomListener(document.getElementById(DrawingManager.buttons[BUTTON_DELETE].id), 'click', () => {
      this.props.enableDrawingMenuButtons();
      this.clickButtonActions(false);
      this.currentButton = BUTTON_DELETE;
      this.deleteAllShape(this.currentButton);
      this.state[DRAWING_MANAGER].setDrawingMode(null);
      this.props.activeDrawing(null);
      this.props.deleteMapShapes(true);
    });

    this.props.onDrawingManagerMounted(this);
    componentDidMount(this, this.state[DRAWING_MANAGER], eventMap);
  }

  componentDidUpdate(prevProps) {
    componentDidUpdate(
      this,
      this.state[DRAWING_MANAGER],
      eventMap,
      updaterMap,
      prevProps,
    );
  }

  componentWillUnmount() {
    componentWillUnmount(this);
    const drawingManager = this.state[DRAWING_MANAGER];
    if (drawingManager) {
      drawingManager.setMap(null);
    }
  }

  clickButtonActions = (areas) => {
    if (!areas) {
      this.props.setL7Confirm(false);
      document.querySelector('#map_holder').classList.remove('areaselect');
    } else {
      this.props.setL7Confirm(true);
      document.querySelector('#map_holder').classList.add('areaselect');
    }
    deselectAllButton(() => {});
    this.props.drawAreaTiles(areas);
    this.props.onCloseCircleLabel();
    this.props.hideOpenInfoBox();    
  }

  activateAreaSelection = () => {
    // const drawingManager = this.state[DRAWING_MANAGER];
    const button = document.getElementById(DrawingManager.buttons[BUTTON_SELECT_AREA].id);
    this.currentButton = BUTTON_SELECT_AREA;
    button.classList.add('drawmanager_button_selected');
    this.props.disableDrawingMenuButtons();
    this.deleteAllShape(this.currentButton);
    // drawingManager.setDrawingMode(google.maps.drawing.OverlayType.MARKER);
  }

  /**
   * Returns the DrawingManager's drawing mode.
   * @type OverlayTypeDrawingManager
   * @public 
   */
  getDrawingMode() {
    return this.state[DRAWING_MANAGER].getDrawingMode();
  }
   
  deleteAllShape(buttonType) {
    this.props.onClearMap(buttonType);
    for (let i = 0; i < this.state[ALL_SHAPE].length; i++) {
      this.state[ALL_SHAPE][i].overlay.setMap(null);
    }
    this.state[ALL_SHAPE] = [];
  }

  render() {
    return null;
  }  
}

export default DrawingManager;
