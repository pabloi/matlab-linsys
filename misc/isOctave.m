function flag=isOctave;
  flag = exist('OCTAVE_VERSION', 'builtin') ~= 0;
end
