function [payoffs] = playGames(ys, antiCoordGame, coordGame)
    payoffs = [ys(1,:) * antiCoordGame * (ys(2,:)' + ys(3,:)'); %alice
               ys(2,:) * antiCoordGame * (ys(1,:)' + ys(3,:)'); %bob
               ys(3,:) * coordGame * (ys(1,:)' + ys(2,:)')]; %eve
    
end