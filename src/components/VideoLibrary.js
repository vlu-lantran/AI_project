import React, { useState } from 'react';

function VideoLibrary({ selectVideo }) {
  const videos = [
    { id: 1, title: 'VIDEO 1', date: '2024-11-09', url: './videos/video1.mp4' },
    { id: 2, title: 'VIDEO 2', date: '2024-11-10', url: './videos/video2.mp4' },
    { id: 3, title: 'VIDEO 3', date: '2024-11-11', url: './videos/video3.mp4' },
    { id: 4, title: 'VIDEO 4', date: '2024-11-12', url: './videos/video4.mp4' },
    { id: 5, title: 'VIDEO 5', date: '2024-11-13', url: './videos/video5.mp4' },
    { id: 6, title: 'VIDEO 6', date: '2024-11-14', url: './videos/video6.mp4' },
    { id: 7, title: 'VIDEO 7', date: '2024-11-15', url: './videos/video7.mp4' },
    { id: 8, title: 'VIDEO 8', date: '2024-11-16', url: './videos/video8.mp4' },
    { id: 9, title: 'VIDEO 9', date: '2024-11-17', url: './videos/video9.mp4' },
    { id: 10, title: 'VIDEO 10', date: '2024-11-18', url: './videos/video10.mp4' },
    { id: 11, title: 'VIDEO 11', date: '2024-11-19', url: './videos/video11.mp4' },
  ];

  const [currentPage, setCurrentPage] = useState(1);
  const videosPerPage = 9;

  // Tính toán số trang và các video trên trang hiện tại
  const totalPages = Math.ceil(videos.length / videosPerPage);
  const indexOfLastVideo = currentPage * videosPerPage;
  const indexOfFirstVideo = indexOfLastVideo - videosPerPage;
  const currentVideos = videos.slice(indexOfFirstVideo, indexOfLastVideo);

  return (
    <div className="video-library">
      <h2>KHO LƯU TRỮ</h2>
      <div className="video-grid">
        {currentVideos.map((video) => (
          <div
            key={video.id}
            className="video-card"
            onClick={() => selectVideo(video)}
          >
            <div className="video-thumbnail">
              <video width="100" height="100" muted>
                <source src={video.url} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            </div>
            <h3>{video.title}</h3>
            <p>Ngày: {video.date}</p>
            <div className="video-buttons">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  selectVideo(video);
                  console.log(`Anomaly Detection: ${video.title}`);
                }}
              >
                Anomaly Detection
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  selectVideo(video);
                  console.log(`Object Detection: ${video.title}`);
                }}
              >
                Object Detection
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  selectVideo(video);
                  console.log(`Car Plate Detection: ${video.title}`);
                }}
              >
                Car Plate Detection
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Điều khiển phân trang */}
      <div className="pagination">
        <button
          onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
          disabled={currentPage === 1}
        >
          Previous
        </button>
        <span>
          Trang {currentPage} / {totalPages}
        </span>
        <button
          onClick={() => setCurrentPage((prev) => Math.min(prev + 1, totalPages))}
          disabled={currentPage === totalPages}
        >
          Next
        </button>
      </div>
    </div>
  );
}

export default VideoLibrary;
