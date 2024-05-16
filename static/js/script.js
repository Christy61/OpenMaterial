class BeforeAfter {
    constructor(entryObject) {
        const beforeAfterContainer = document.querySelector(entryObject.id);
        const before = beforeAfterContainer.querySelector('.bal-before');
        const middle = beforeAfterContainer.querySelector('.bal-middle');
        const after = beforeAfterContainer.querySelector('.bal-after');
        const handle1 = document.getElementById(entryObject.handle1);
        const handle2 = document.getElementById(entryObject.handle2);
        const beforeLabel = beforeAfterContainer.querySelector('.beforeLabel');
        const middleLabel = beforeAfterContainer.querySelector('.middleLabel');
        const afterLabel = beforeAfterContainer.querySelector('.afterLabel');

        const updateClipPaths = () => {
            const containerWidth = beforeAfterContainer.offsetWidth;
            const handle1Pos = handle1.offsetLeft + handle1.offsetWidth / 2;
            const handle2Pos = handle2.offsetLeft + handle2.offsetWidth / 2;

            before.style.clipPath = `inset(0 ${containerWidth - handle1Pos}px 0 0)`;
            middle.style.clipPath = `inset(0 ${containerWidth - handle2Pos}px 0 ${handle1Pos}px)`;
            after.style.clipPath = `inset(0 0 0 ${handle2Pos}px)`;

            middleLabel.style.left = `${handle1Pos}px`;
            afterLabel.style.left = `${handle2Pos}px`;
        };

        const handleDrag = (handle, callback) => {
            let startX = 0;
            let startLeft = 0;

            const onMouseMove = (e) => {
                const containerRect = beforeAfterContainer.getBoundingClientRect();
                const newLeft = startLeft + (e.clientX - startX);

                if (newLeft > 0 && newLeft < containerRect.width) {
                    callback(newLeft);
                }
            };

            const onMouseUp = () => {
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);
            };

            const onMouseDown = (e) => {
                startX = e.clientX;
                startLeft = handle.offsetLeft;
                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', onMouseUp);
            };

            handle.addEventListener('mousedown', onMouseDown);
        };

        handleDrag(handle1, (newLeft) => {
            handle1.style.left = `${newLeft - handle1.offsetWidth / 2}px`;
            updateClipPaths();
        });

        handleDrag(handle2, (newLeft) => {
            handle2.style.left = `${newLeft - handle2.offsetWidth / 2}px`;
            updateClipPaths();
        });

        beforeAfterContainer.addEventListener("touchstart", (e) => {
            const handle = e.target.closest('.bal-handle');
            if (!handle) return;

            const startX = e.touches[0].clientX;
            const startLeft = handle.offsetLeft;

            const onTouchMove = (e2) => {
                const containerRect = beforeAfterContainer.getBoundingClientRect();
                const newLeft = startLeft + (e2.touches[0].clientX - startX);

                if (newLeft > 0 && newLeft < containerRect.width) {
                    handle.style.left = `${newLeft - handle.offsetWidth / 2}px`;
                    updateClipPaths();
                }
            };

            const onTouchEnd = () => {
                beforeAfterContainer.removeEventListener('touchmove', onTouchMove);
                beforeAfterContainer.removeEventListener('touchend', onTouchEnd);
            };

            beforeAfterContainer.addEventListener('touchmove', onTouchMove);
            beforeAfterContainer.addEventListener('touchend', onTouchEnd);
        });

        window.addEventListener('resize', updateClipPaths);
        updateClipPaths();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    for (let i = 1; i <= 4; i++) {
        new BeforeAfter({
            id: `#example${i}`,
            handle1: `handle1-${i}`,
            handle2: `handle2-${i}`
        });
    }
});
