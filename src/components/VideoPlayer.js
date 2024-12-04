import React, { useEffect, useRef } from 'react';

function VideoPlayer({ video, detectionType, goBack }) {
  const videoRef = useRef(null);

  useEffect(() => {
    if (video && videoRef.current) {
      videoRef.current.play();
    }
  }, [video]);

  const handleDownloadJSON = () => {
    const jsonData = {
      id: video.id,
      title: video.title,
      date: video.date,
      detectionType: detectionType || 'Unknown',
    };

    const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${video.title}_info.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="video-player">
      {video ? (
        <>
          <video ref={videoRef} width="80%" height="auto" controls>
            <source src={video.url} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
          <div className="video-info">
            <h2>{video.title}</h2>
            <p><strong>Ngày:</strong> {video.date}</p>
            <p><strong>Tải File JSON:</strong> {detectionType}</p>
          </div>
          <button onClick={handleDownloadJSON}>Download JSON</button>
        </>
      ) : (
        <p>No Videos Selected.</p>
      )}
    </div>
  );
}

export default VideoPlayer;
