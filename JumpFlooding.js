// COMPUTE THE DIGITAL VORONOI DIAGRAM VIA JUMP FLOODING ALGORITHM

const {
    GPUComposer,
    GPULayer,
    GPUProgram,
    renderAmplitudeProgram,
    renderSignedAmplitudeProgram,
    multiplyValueProgram,
    FLOAT,
    INT,
    REPEAT,
    CLAMP_TO_EDGE,
    NEAREST,
  } = GPUIO ;
  
  // Init a canvas element.
  const canvas = document.createElement('canvas');
  document.body.appendChild(canvas);
  // Init a composer.
  const composer = new GPUComposer({ canvas });
  
  // PARAMETERS
  nPts = 10;
  textureSize = [canvas.width,canvas.height];
  textureSize.forEach((el, i) => textureSize[i] = 2**(math.floor(math.log2(textureSize[i]))+0));
  jump_step = math.multiply(textureSize,1); // initial jump distance
  pts_margin = 0/math.sqrt(nPts);
  clr_margin = .1;

  // List of random cell center points & colors
  points = []
  colors = []
  for (let p=0;p<nPts;p++){
    //points.push([math.random()*textureSize[0],math.random()*textureSize[1]])
    points[p] = [math.random(),math.random()];
    //points[p] = math.add(math.multiply([points[p][0]*math.cos(2*3.1416*points[p][1]),points[p][0]*math.sin(2*3.1416*points[p][1])],.5),.5)
    points[p] = math.add(math.multiply(points[p],1-2*pts_margin),pts_margin);
    colors[p] = [math.random(),math.random(),math.random(),1]
    colors[p] = math.add(math.multiply(colors[p],1-2*clr_margin),clr_margin);
  }
  const centers = new GPULayer(composer, {
    name: 'centers',
    dimensions: [nPts,1],
    numComponents: 2, // Scalar voronoi has one component.
    type: FLOAT,
    filter: NEAREST,
    // Use 2 buffers so we can toggle read/write
    // from one to the other.
    numBuffers: 2,
    wrapX: CLAMP_TO_EDGE,
    wrapY: CLAMP_TO_EDGE,
    array: points.flat(),
  });

  const clr = new GPULayer(composer, {
    name: 'colors',
    dimensions: [nPts,1],
    numComponents: 4, // Scalar voronoi has one component.
    type: FLOAT,
    filter: NEAREST,
    // Use 2 buffers so we can toggle read/write
    // from one to the other.
    numBuffers: 1,
    wrapX: CLAMP_TO_EDGE,
    wrapY: CLAMP_TO_EDGE,
    array: colors.flat(),
  });
  
  // Init the voronoi texture.
  const particles = new Float32Array(textureSize[0]*textureSize[1]).fill(-1);
  for (let p=0;p<nPts;p++) {
    let x = points[p];
    //let x = math.floor(points[p]);
    let i = (x[0] + x[1]*textureSize[1])*textureSize[0];
    //i = x[0] + x[1]*textureSize[0];
    particles[math.floor(i)] = p/nPts;
  }

  const voronoi = new GPULayer(composer, {
    name: 'voronoi',
    dimensions: textureSize,
    numComponents: 1, // Scalar voronoi has one component.
    type: FLOAT,
    filter: NEAREST,
    // Use 2 buffers so we can toggle read/write
    // from one to the other.
    numBuffers: 2,
    wrapX: CLAMP_TO_EDGE,
    wrapY: CLAMP_TO_EDGE,
    array: particles,
  });

  const distance = new GPULayer(composer, {
    name: 'distance',
    dimensions: textureSize,
    numComponents: 1, // Scalar voronoi has one component.
    type: FLOAT,
    filter: NEAREST,
    // Use 2 buffers so we can toggle read/write
    // from one to the other.
    numBuffers: 2,
    wrapX: CLAMP_TO_EDGE,
    wrapY: CLAMP_TO_EDGE,
    array: new Float32Array(textureSize[0]*textureSize[1]).fill(-1),
  });
  
  // Init a program to diffuse voronoi.
  const growthProgram = new GPUProgram(composer, {
    name: 'growth',
    fragmentShader: `
      in vec2 v_uv;
  
      uniform sampler2D u_voronoi;
      uniform sampler2D u_centers;
      uniform vec2 u_step;
  
			layout (location = 0) out float particle_idx; // Output at index 0.
			layout (location = 1) out float distance; // Output at index 1.
  
      void main() {

      // Initialize
        particle_idx = texture(u_voronoi, v_uv).x;
        vec2 particle_pos = texture(u_centers,vec2(particle_idx,0.0)).xy;
        if (particle_idx<0.0) distance = 1000000.0 ;
        else distance = length(particle_pos-v_uv) ;

      // Loop over neightboring pixels
        for (float i=-1.0;i<=1.0;i++){
          for (float j=-1.0;j<=1.0;j++) {
            if (abs(j)+abs(i)<1.0) continue; // do not test the current position
            //if (abs(j)+abs(i)>=2.0) continue;

          // UV coordinate of the neightbor
            vec2 neightbor_uv = v_uv + vec2(i*u_step.x,j*u_step.y);
          // Associated particle index
            float neightbor_idx = texture(u_voronoi, neightbor_uv).x;
            if (neightbor_idx<0.0) continue; // the neightbor is not valid
          // Correponding particle position & distance
            vec2 neightbor_pos = texture(u_centers,vec2(neightbor_idx,0.0)).xy;
            float neightbor_distance = length(neightbor_pos-v_uv);

          // If the current point is not valid & the neightbor is valid..
            if (particle_idx<0.0) { // the current point is not valid and the neightbor is valid
              particle_idx = neightbor_idx;
              distance = neightbor_distance;
              continue;
            }

          // If this point was already valid..
            if (neightbor_distance<distance) {
              particle_idx = neightbor_idx;
              distance = neightbor_distance ;
            }
          }
        }

      // After all this..
      //distance = length(particle_pos-v_uv);
      }
    `,
    uniforms: [
      { // Index of sampler2D uniform to assign to value "u_voronoi".
        name: 'u_voronoi',
        value: 0,
        type: INT,
      },
      { // Index of sampler2D uniform to assign to value "u_voronoi".
        name: 'u_centers',
        value: 1,
        type: INT,
      },
      { // Calculate the size of a 1 px step in UV coordinates.
        name: 'u_step',
        value: [1 / textureSize[0], 1 / textureSize[1]],
        type: FLOAT,
      },
    ],
  });
  
  // Init a program to diffuse voronoi.
  const floodProgram = new GPUProgram(composer, {
    name: 'flood',
    fragmentShader: `
      in vec2 v_uv;
  
      uniform sampler2D u_voronoi;
      uniform vec2 u_step;
  
      out float value;
  
      void main() {
        value = texture(u_voronoi, v_uv).x;
        if (value>=0.0) return;
        value = texture(u_voronoi, v_uv + vec2(0, u_step.y)).x;
        if (value>=0.0) return;
        value = texture(u_voronoi, v_uv - vec2(0, u_step.y)).x;
        if (value>=0.0) return;
        value = texture(u_voronoi, v_uv + vec2(u_step.x, 0)).x;
        if (value>=0.0) return;
        value = texture(u_voronoi, v_uv - vec2(u_step.x, 0)).x;
      }
    `,
    uniforms: [
      { // Index of sampler2D uniform to assign to value "u_voronoi".
        name: 'u_voronoi',
        value: 0,
        type: INT,
      },
      { // Calculate the size of a 1 px step in UV coordinates.
        name: 'u_step',
        value: [1 / textureSize[0], 1 / textureSize[1]],
        type: FLOAT,
      },
    ],
  });

  // Init a program to diffuse voronoi.
  const renderVoronoiProgram = new GPUProgram(composer, {
    name: 'apply_color',
    fragmentShader: `
      in vec2 v_uv;
  
      uniform sampler2D u_voronoi;
      uniform sampler2D u_colors;
      uniform vec2 u_step;
  
      out vec4 value;
  
      void main() {
        float particle_idx = texture(u_voronoi,v_uv).x;
        if (particle_idx<0.0) value = vec4(0.0);
        else value = texture(u_colors,vec2(particle_idx,0.0));
      }
    `,
    uniforms: [
      { // Index of sampler2D uniform to assign to value "u_voronoi".
        name: 'u_voronoi',
        value: 0,
        type: INT,
      },
      { // Index of sampler1D uniform to assign to value "u_colors".
        name: 'u_colors',
        value: 1,
        type: INT,
      },
    ],
  });

  // Init a program to diffuse voronoi.
  const renderDistanceProgram = renderSignedAmplitudeProgram(composer, {
    name: 'render_distance',
    type: distance.type,
    components: 'x',
  });
  
  // Simulation/render loop.
  function loop() {
  
    // Diffuse voronoi and write result to voronoi.
    for (let it=0;it<1;it++) 
    {
    // Grow voronoi cells
      growthProgram.setUniform('u_step',[jump_step[0]/textureSize[0],jump_step[1]/textureSize[1]]);
      composer.step({
        program: growthProgram,
        input: [voronoi,centers],
        output: [voronoi,distance],
      });
      // Divide the jump by two
      jump_step[0]=math.max(jump_step[0]/2,1);
      jump_step[1]=math.max(jump_step[1]/2,1);
    }
  
    // If no "output", will draw to canvas.
    composer.step({
      program: renderVoronoiProgram,
      input: [voronoi,clr],
    });
    // composer.step({
    //   program: renderDistanceProgram,
    //   input: distance
    // });

    window.requestAnimationFrame(loop);
  }
  loop(); // Start animation loop.