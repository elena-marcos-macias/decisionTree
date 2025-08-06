function makeSquare(ch)
    % Adjust the position of the confusion chart to make it square
    drawnow;  % Ensure chart is fully rendered
    pos = ch.Position;

    % Make it square inside the tile
    side = min(pos(3), pos(4));  % take the smaller of width or height
    centerX = pos(1) + pos(3)/2;
    centerY = pos(2) + pos(4)/2;

    % Re-center the square chart
    ch.Position = [centerX - side/2, centerY - side/2, side, side];
end