// Photo credits. Fill in photographer, URLs and license when the photos are added
// to src/assets/photos/ (photo-a.jpg, photo-b.jpg) and mirror them in CREDITS.md.
export interface Credit {
  where: string;
  file: string;
  description: string;
  photographer?: string;
  photographerUrl?: string;
  sourceUrl?: string;
  license?: string;
}

export const CREDITS: Credit[] = [
  {
    where: 'Problem section (Photo A)',
    file: 'src/assets/photos/photo-a.jpg',
    description: 'A health worker or clinic scene in a low- or middle-income country, natural light.',
  },
  {
    where: 'About section (Photo B)',
    file: 'src/assets/photos/photo-b.jpg',
    description: 'Students working together on a laptop.',
  },
  {
    where: 'About section (founder portrait)',
    file: 'src/assets/brand/founder.png',
    description: 'Portrait of Aayush Dinker, supplied by the founder.',
  },
];
