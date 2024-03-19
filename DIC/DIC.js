
const {
  GPUComposer,
  GPULayer,
  GPUProgram,
  copyProgram,
  renderRGBProgram,
  renderAmplitudeProgram,
  renderSignedAmplitudeProgram,
  FLOAT,
  INT,
  REPEAT,
  CLAMP_TO_EDGE,
  LINEAR,
  NEAREST,
} = GPUIO ;

// GLOBAL PARAMETERS
const PARAMS = {
  DIC: {
    active: true,
    decim: 10,
    winSZ: 15,
    iter: 1, // number of iterations by frame
  },
  RefFrame: {
    defined: false, // is the reference frame defined ?
    usePrevious: true, // use previous frame as reference ?
  },
}

// Init a tweakpane
const pane = new Tweakpane.Pane({title:'Digital Image Correlation'});
pane.registerPlugin(TweakpaneEssentialsPlugin);
const fpsGraph = pane.addBlade({view: 'fpsgraph',label: 'fpsgraph',});
f = pane.addFolder({title:"DIC"}) ;
  f.addInput(PARAMS.DIC,'active');
  f.addInput(PARAMS.DIC,'decim',{min:1,max:20,step:1})
    .on('change',(e) => {position.resize([math.floor(videoWidth/e.value),math.floor(videoHeight/e.value)]);});
  f.addInput(PARAMS.DIC,'winSZ',{min:1,max:40,step:1})
    .on('change',(e) => DICProgram.setUniform('u_win_sz',PARAMS.DIC.winSZ));
  f.addInput(PARAMS.DIC,'iter',{min:1,max:10,step:1});
f = pane.addFolder({title:"Ref Frame"}) ;
  f.addInput(PARAMS.RefFrame,'usePrevious')
  f.addButton({title:"Reset"})
    .on('click',(e) => PARAMS.RefFrame.defined=false);


// Init the canvas
const canvas = document.createElement('canvas');
var gl = canvas.getContext('webgl2');
document.body.appendChild(canvas);

// GRAB THE VIDEO OBJECT
// var videoWidth = 320; var videoHeight = 240;
var videoWidth = 640; var videoHeight = 480;
//var videoWidth = 1080; var videoHeight = 720;
videoWidth = math.pow(2,math.floor(math.log2(videoWidth))); videoHeight = math.pow(2,math.floor(math.log2(videoHeight)));
var videoResolution = [videoWidth, videoHeight];
var videoReady = false;
var video = null;
setupVideo();
async function setupVideo() {
  if(video == null) {
      video = document.createElement("video");
      let stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: videoWidth, height: videoHeight },
        audio: false, 
      });
      video.srcObject = stream;
      video.muted = true;
      video.playsInline = true;
      video.onloadedmetadata = function(e) {
          video.play();
          videoReady = true;
      };
  }
}
var lastTime = -1;

// Init a composer.
const composer = new GPUComposer({ canvas });

// COMMON GPU LAYER PARAMETERS
const GPULayerParams = {
  dimensions: videoResolution,
  type: FLOAT,
  filter: NEAREST,
  wrapX: CLAMP_TO_EDGE,
  wrapY: CLAMP_TO_EDGE,
}

// THE CAMERA GPU LAYER
const camera = new GPULayer(composer, Object.assign({},GPULayerParams,{
  name: 'camera',
  numComponents: 3, // RGB Color
  numBuffers: 1,
}));
const renderColorProgram = renderRGBProgram(composer, {
  name: 'renderColor',
  type: camera.type,
});

//----------------------------------------------------------
// RAW MONOCHROME FRAMES
//----------------------------------------------------------
const frames = new GPULayer(composer, Object.assign({},GPULayerParams,{
  name: 'frames',
  numComponents: 1, // monochrome
  numBuffers: 2, // currentFrame, previousFrame
  filter: LINEAR,
}));
const previousFrame = new GPULayer(composer, Object.assign({},GPULayerParams,{
  name: 'previousFrame',
  numComponents: 1, // monochrome
  numBuffers: 1,
}));
const RGB2MonoProgram = new GPUProgram(composer, {
  name: 'monochrome',
  fragmentShader: `
    in vec2 v_uv; uniform sampler2D u_RGB; out float out_mono;
    void main() {out_mono = (texture(u_RGB,v_uv).r + texture(u_RGB,v_uv).g + texture(u_RGB,v_uv).b)/3.0;}
  `,
  uniforms: [ {name: 'u_RGB',value: 0,type: INT,}, ],
});
const copyFrameProgram = copyProgram(composer,{
  type: frames.type,
})
const renderMonoProgram = renderAmplitudeProgram(composer, {
  name: 'renderMonochrome',
  type: frames.type,
});
const renderComposeProgram = new GPUProgram(composer, {
  name: 'compose_frames',
  fragmentShader: `
    in vec2 v_uv; uniform sampler2D u_current; uniform sampler2D u_ref; out vec4 out_comp;
    void main() {
      float current = texture(u_current,v_uv).x;
      float ref = texture(u_ref,v_uv).x;
      out_comp = vec4(current,(current+ref)/2.0,ref,1.0);
    }
  `,
  uniforms: [ {name: 'u_current',value: 0,type: INT,}, ],
  uniforms: [ {name: 'u_ref',value: 1,type: INT,}, ],
});


//----------------------------------------------------------
// FRAME GRADIENT
//----------------------------------------------------------
const gradient = new GPULayer(composer, Object.assign({},GPULayerParams,{
  name: 'gradient',
  numComponents: 2, // 2D gradient
  numBuffers: 1,
}));
const gradientProgram = new GPUProgram(composer, {
  name: 'gradient',
  fragmentShader: `
    in vec2 v_uv;
    uniform sampler2D u_frame;
    uniform vec2 u_step;
    out vec2 out_grad;

    float der(sampler2D fr, vec2 step) {
      float d = 0.5*(texture(fr,v_uv+step).x - texture(fr,v_uv-step).x);
      return d;
    }

    void main() {
        float dfx = der(u_frame, vec2(u_step.x,0.0));
        float dfy = der(u_frame, vec2(0.0,u_step.y));
        out_grad = vec2(dfx, dfy);
    }
  `,
  uniforms: [
    { name: 'u_frame', value: 0, type: INT, }, // the frame to derive
    { name: 'u_step', value: [1/videoWidth,1/videoHeight], type: FLOAT, }, // the pixel step size in tex coordinates
  ],
});



//----------------------------------------------------------
// DIC
//----------------------------------------------------------
const position = new GPULayer(composer, Object.assign({},GPULayerParams,{
  name: 'position',
  numComponents: 2, // 2D position
  numBuffers: 2, // allows position updating
  clearValue: [0,0],
  dimensions: [math.floor(videoWidth/PARAMS.DIC.decim),math.floor(videoHeight/PARAMS.DIC.decim)],
}));
const DICProgram = new GPUProgram(composer, {
  name: 'DIC',
  fragmentShader: `
    in vec2 v_uv;
    uniform sampler2D u_ref_frame;
    uniform sampler2D u_current_frame;
    uniform sampler2D u_grad;
    uniform sampler2D u_position;
    uniform vec2 u_step;
    uniform int u_win_sz;
    out vec2 out_pos;

    void main() {
      vec2 pos = u_step*texture(u_position,v_uv).xy;
      vec2 j = vec2(0.0); // jacobian (j1,j2)
      vec3 H = vec3(0.0); // Hessian (H11,H22,H12)
      vec2 hsz = 0.5*float(u_win_sz)*u_step; // window half size
      float dx = -hsz.x;
      for (int ddx=0; ddx<u_win_sz; ddx++)
      {
        dx+=u_step.x; 
        float dy = -hsz.y;
        for (int ddy=0; ddy<u_win_sz; ddy++)
        {
          dy+=u_step.y; 
          vec2 iv_uv_ref = v_uv + vec2(dx,dy) ;
          vec2 iv_uv_current = iv_uv_ref + pos ;
          vec2 g = texture2D(u_grad,iv_uv_current).xy ;
          float d = texture(u_current_frame,iv_uv_current).x-texture(u_ref_frame,iv_uv_ref).x ;
          j += d*g;
          H += vec3(g.x*g.x,g.y*g.y,g.x*g.y);
        }
      }
      float idetH = 1.0/(H.x*H.y-H.z*H.z);
      vec3 iH = idetH*vec3(H.y,H.x,-H.z); // (iH11,iH22,iH12)
      out_pos = pos/u_step - vec2( iH.x*j.x + iH.z*j.y, iH.z*j.x + iH.y*j.y );
    }
  `,
  uniforms: [
    { name: 'u_current_frame', value: 0, type: INT, },
    { name: 'u_ref_frame', value: 1, type: INT, },
    { name: 'u_grad', value: 2, type: INT, },
    { name: 'u_position', value: 3, type: INT, },
    { name: 'u_step', value: [1/videoWidth, 1/videoHeight], type: FLOAT, },
    { name: 'u_win_sz', value: PARAMS.DIC.winSZ, type: INT, },
  ],
});


//----------------------------------------------------------
// RENDER LOOP
//----------------------------------------------------------
function loop() {
  window.requestAnimationFrame(loop);

  fpsGraph.begin();


  // Copy the video content to the camera layer
  if(videoReady && !video.paused) {
  // Is there a new frame to process ?
    let time = video.currentTime;
    if (time==lastTime) return;
    lastTime = time;
  // Copy in camera buffer
    gl.bindTexture(gl.TEXTURE_2D, camera._buffers[camera._bufferIndex]);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, video);
    gl.generateMipmap(gl.TEXTURE_2D);
  }

  // Reference frame
  if (PARAMS.RefFrame.defined && !PARAMS.RefFrame.usePrevious) { 
    frames.decrementBufferIndex(); // prevents the change of frames.lastState
  }
  if (!PARAMS.RefFrame.defined || PARAMS.RefFrame.usePrevious) position.clear() ;

  // Push monochrome images
  composer.step({
    program: RGB2MonoProgram, 
    input: camera, 
    output: frames, // will increment frames's bufferindex
  });
  PARAMS.RefFrame.defined = true;

  // If no "output", will draw to canvas.
  composer.step({
    // program: renderColorProgram, input: camera,
    // program: renderMonoProgram, input: frames.currentState
    program: renderComposeProgram, input: [frames.currentState,frames.lastState]
  });

  if (PARAMS.DIC.active) {
    // Image gradient
    composer.step({
      program: gradientProgram, 
      input: frames.currentState,
      output: gradient,
    });

    // DIC
    for (let it=0;it<PARAMS.DIC.iter;it++)
      composer.step({
        program: DICProgram, 
        input: [frames.currentState,frames.lastState,gradient,position], // does not increment frame buffer
        output: position,
      });

    composer.drawLayerAsVectorField({
      layer: position,
      vectorSpacing: PARAMS.DIC.decim,
      vectorScale: 2,
      color: [1, 1, 1],
    });
  }

  fpsGraph.end();

}
loop(); // Start animation loop.